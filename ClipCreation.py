import numpy as np
import torch
import os

from scipy.ndimage import distance_transform_edt

from torch.utils.data import DataLoader, SequentialSampler

from Models.FDU3D import *

from tqdm import tqdm
import argparse

from era5dataset.FrontDataset import *
# ERA Extractors
from era5dataset.EraExtractors import *
from IOModules.csbReader import *
from NetInfoImport import *
from FrontPostProcessing import filterFronts

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# Current Best
# Medium Bottle Net, 32 Batchsize, BottleneckLayer 128 256 128, 3 levels, lr = 0.01, lines +- 1
# ~ 45% validation loss 

from skimage import measure, morphology
from InferOutputs import inferResults, setupDataLoader, setupDevice, setupModel


import netCDF4

from FrontalCrossSection import getTgtRange, getDate, getSecondaryFile, getSecondaryData

from era5dataset.ERA5Reader.readNetCDF import getValueRanges
import imageio
from moviepy.editor import VideoFileClip


def parseArguments():
    parser = argparse.ArgumentParser(description='FrontNet')
    parser.add_argument('--net', help='path no net')
    parser.add_argument('--data', help='path to folder containing data')
    parser.add_argument('--outname', help='name of the output')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--device', type = int, default = 0, help = "number of device to use")
    parser.add_argument('--fullsize', action='store_true', help='test the network at the global scope')
    parser.add_argument('--preCalc', action='store_true', help='test the network at the global scope')
    parser.add_argument('--NWS', action = 'store_true', help='use Resolution of hires')
    parser.add_argument('--num_samples', type = int, default = -1, help='number of samples to infere from the dataset')
    parser.add_argument('--classes', type = int, default = 1, help = 'How many classes the network should predict (binary case has 1 class denoted by probabilities)')
    parser.add_argument('--normType', type = int, default = 0, help = 'How to normalize the data: 0 min-max, 1 mean-var, 2/3 the same but per pixel')
    parser.add_argument('--labelGroupingList', type = str, default = None, help = 'Comma separated list of label groups \n possible fields are w c o s (warm, cold, occluson, stationary)')
    parser.add_argument('--ETH', action = 'store_true', help = 'Compare against an ETH result instead of net')
    parser.add_argument('--show-error', action = 'store_true', help = 'show the inividual error values during inference')
    parser.add_argument('--fromFile', type = str, default = None, help = 'show the inividual error values during inference')
    parser.add_argument('--calcVar', type = str, default = "t", help = 'which variable to measure along the cross section')
    parser.add_argument('--secPath', type = str, default = None, help = 'Path to folder with secondary data containing variable information to be evaluated. Data should be stored as <secPath>/YYYY/MM/<fileID>YYYYMMDD_HH.nc . <fileID> is an Identifier based on the type of file (e.g. B,Z,precip)')
    parser.add_argument('--alpha', type = float, default = 0, help='weight of constant background compared background variable. [0 to 1]')
    parser.add_argument('--rgb', nargs=3, type = int, help='rgb weights of for the background variable [0..255] x 3')
    parser.add_argument('--lsm', default = None, help='path to land-sea-mask netCDF file')
    args = parser.parse_args()

    args.binary = args.classes == 1
    
    return args
    
def setupDataset(args):
    data_fold = args.data
    stepsize = 0.25
    if(args.fullsize):
        cropsize = (720, 1440)
        mapTypes = {"NA": ("NA", (90,-89.75), (-180,180), (-0.25,0.25)) }
        if(args.NWS):
            mapTypes = {"hires": ("hires", (90, -89.75), (-180, 180), (-0.25,0.25)) }
    elif(args.preCalc and not args.fullsize):
        # add another 5 degree to the input, such that we can savely extract lines from fronts at the corner of evaluation
        cropsize = (55*4,100*4)
        mapTypes = {"NA": ("NA", (75+5,30.25-5), (-50-5,40+5), (-stepsize, stepsize), None)}
        if(args.NWS):
            cropsize = (55*4, 95*4) 
            mapTypes = {"hires": ("hires", (75+5, 30.25-5), (-140-5, -55+5), (-stepsize,stepsize), None) }
    else:
        # add another 5 degree to the input, such that we can savely extract lines from fronts at the corner of evaluation
        cropsize = (56*4,100*4)
        mapTypes = {"NA": ("NA", (76+5,30.25-5), (-50-5,40+5), (-stepsize, stepsize), None)}
        if(args.NWS):
            cropsize = (56*4, 96*4) 
            mapTypes = {"hires": ("hires", (76+5, 30.25-5), (-141-5, -55+5), (-stepsize,stepsize), None) }
    
    myLevelRange = np.arange(105,138,4)

    myTransform = (None, None)
    labelThickness = 1
    labelTrans = (0,0)

    labelGroupingList = args.labelGroupingList
    myLineGenerator = extractStackedPolyLinesInRangeAsSignedDistance(labelGroupingList, labelThickness, labelTrans)
    myLabelExtractor = DefaultFrontLabelExtractor(myLineGenerator)

    variables = ['t','q','u','v','w','sp','kmPerLon']

    normType = args.normType

    if(not args.fromFile is None):
        info = getDataSetInformationFromInfo(args.fromFile)
        print(info)
        variables = info["Variables"]
        normType = info["NormType"]
        myLevelRange = info["levelrange"]
        print(variables, normType, myLevelRange)

    myEraExtractor = DerivativeFlippingAwareEraExtractor(variables, [], [], 0.0, 0 , 1, normType = normType, sharedObj = None)

    subfolds = (True, False)
    remPref = 3

    if(ETH):
        myEraExtractor = ETHEraExtractor()
        subfolds = (False, False)
        remPref = 1
    if(args.preCalc):
        myEraExtractor = BinaryResultExtractor()
        subfolds = (False, False)
        remPref = 0
    

    # Create Dataset
    data_set = WeatherFrontDataset(data_dir=data_fold, mapTypes = mapTypes, levelRange = myLevelRange, transform=myTransform, outSize=cropsize, labelThickness= labelThickness, label_extractor = myLabelExtractor, era_extractor = myEraExtractor, has_subfolds = subfolds, asCoords = False, removePrefix = remPref)
    return data_set


def readSecondary(rootgrp, var, time, level, latrange, lonrange):
    vals = np.zeros((abs(int((latrange[0]-latrange[1])*4)), abs(int((lonrange[1]-lonrange[0])*4))))
    if(level is None):
        if(lonrange[0] < 0 and lonrange[1] >= 0):
            vals[:,:-int(lonrange[0]*4)] =  rootgrp[var][time,int((90-latrange[0])*4):int((90-latrange[1])*4), int(lonrange[0]*4):]
            vals[:,-int(lonrange[0]*4):] = rootgrp[var][time,int((90-latrange[0])*4):int((90-latrange[1])*4), :int(lonrange[1]*4)]
        else:
            vals[:,:] = rootgrp[var][time,int((90-latrange[0])*4):int((90-latrange[1])*4), int(lonrange[0]*4):int(lonrange[1]*4)]
    else:
        if(lonrange[0] < 0 and lonrange[1] >= 0):
            vals[:,:-int(lonrange[0]*4)] =  rootgrp[var][time,level,int((90-latrange[0])*4):int((90-latrange[1])*4), int(lonrange[0]*4):]
            vals[:,-int(lonrange[0]*4):] = rootgrp[var][time,level,int((90-latrange[0])*4):int((90-latrange[1])*4), :int(lonrange[1]*4)]
        else:
            vals[:,:] = rootgrp[var][time,level,int((90-latrange[0])*4):int((90-latrange[1])*4), int(lonrange[0]*4):int(lonrange[1]*4)]
    return vals


def performInference(model, loader, num_samples, parOpt, args):
    
    outfold = os.path.join("Clips", args.outname)
    if not os.path.isdir(outfold):
        os.mkdir(outfold)
    outname = os.path.join(outfold, "clip")
    out_channels = 4
    border = 20
    skip = 0

    # Path to Secondary File
    secondaryPath = args.secPath
    if(secondaryPath is None):
        print("Secondary Path needed for this type of evaluation!")
        exit(1)
    
    data_set = loader.dataset
    no = data_set.removePrefix
    # Get Range
    mapType = "hires" if args.NWS else "NA"
    latoff= (data_set.mapTypes[mapType][1][0]-90)/data_set.mapTypes[mapType][3][0]
    lonoff= (data_set.mapTypes[mapType][2][0]+180)/data_set.mapTypes[mapType][3][1]
    tgt_latrange, tgt_lonrange = getTgtRange(data_set, mapType)
    print("offsets for the corresponding mapType, to estimate distance in km:", tgt_latrange, tgt_lonrange)
    bgFile = args.lsm
    noBg = bgFile is None or (not os.path.isfile(args.lsm))
    if(not noBg):
        bgroot = netCDF4.Dataset(os.path.realpath(bgFile), "r", format="NETCDF4", parallel=False)
        bgMap = (readSecondary(bgroot, "lsm", 0, None, tgt_latrange, tgt_lonrange)>0.0 )*1.0
        contourMap = torch.from_numpy(1 - (morphology.binary_dilation(bgMap)-bgMap))
    else:
        contourMap = 0
        print("could not find File: ", args.lsm)
    writer = imageio.get_writer(outname+".gif", mode="I", duration=0.1)
    temporalList = np.zeros((num_samples, 5, 8, 8))
    for idx, data in enumerate(tqdm(loader, desc ='eval'), 0):
        if idx<skip:
            continue
        if(idx == num_samples+skip):
            break
        if(not torch.cuda.is_available()):
            inputs, labels, filename = data.data, data.labels, data.filenames
        else:
            inputs, labels, filename = data
            inputs = inputs.to(device = parOpt.device, non_blocking=False)
        
        # remove all short labels (# 1 is a dummy value, evaluation will skip 40 pixel anyways)
        if(args.ETH):
            outputs = inputs.permute(0,2,3,1)
        elif(args.preCalc):
            outputs = (inputs*1).cpu().numpy()
            print(outputs.shape)
        else:
            tgtIn = torch.cat((inputs[:,:6*9], inputs[:,-1:]), dim = 1)
            outputs = inferResults(model, tgtIn, args)
        

        # Prepare the secondary data
        year,month,day,hour = getDate(filename[0], no)  

        # we do not have the 29th of february for ZFiles
        if("_z" in args.calcVar and month == "02" and day == "29"):
            continue
        
        newFile = getSecondaryFile(args.calcVar, secondaryPath, year, month, day, hour)
        var = getSecondaryData(newFile, args.calcVar, tgt_latrange, tgt_lonrange)
        
        # reshuffle input for display
        #switch channels for usual output colors
        outpred = np.zeros((outputs.shape[1],outputs.shape[2],3))
        # red -> warm
        outpred[:,:,0] = outputs[0,:,:,1]
        # green -> stationary
        outpred[:,:,1] = outputs[0,:,:,4]
        # blue -> cold
        outpred[:,:,2] = outputs[0,:,:,2]
        # pink -> occlusion
        outpred[:,:,0] = (outpred[:,:,0]<=outputs[0,:,:,3])*outputs[0,:,:,3] + (outpred[:,:,0] > outputs[0,:,:,3])*outpred[:,:,0]

        outpred[:,:,2] = (outpred[:,:,2]<=outputs[0,:,:,3])*outputs[0,:,:,3] + (outpred[:,:,2] > outputs[0,:,:,3])*outpred[:,:,2]
        # white -> no clear distinction
        # get all Zeros
        zeros = np.nonzero((np.sum(outpred, axis=-1)==0)*1.0)
        # at all zero positins write a yellow line if the general Front is identified
        outpred[zeros[0],zeros[1],0] = outputs[0,zeros[0],zeros[1],0]
        outpred[zeros[0],zeros[1],1] = outputs[0,zeros[0],zeros[1],0]
        #outpred[zeros[0],zeros[1],2] = outputs[0,zeros[0],zeros[1],0]
        
        # normalize it to 0..1
        mini, maxi = getValueRanges(args.calcVar.split("_")[0])
        var = (var - mini) / (maxi-mini)
        mygifImg = outpred
        sumimg = np.sum(mygifImg, axis = -1) < 0.5
        # if a front is present => print the front, else print the background variable
        alpha = args.alpha
        mygifImg[:,:,0] = (alpha*contourMap + (1-alpha) * var * (args.rgb[0]/255.0)) * sumimg + mygifImg[:,:,0] * (~sumimg) 
        mygifImg[:,:,1] = (alpha*contourMap + (1-alpha) * var * (args.rgb[1]/255.0)) * sumimg + mygifImg[:,:,1] * (~sumimg)
        mygifImg[:,:,2] = (alpha*contourMap + (1-alpha) * var * (args.rgb[2]/255.0)) * sumimg + mygifImg[:,:,2] * (~sumimg)
        
        # add the image to the gif
        writer.append_data((mygifImg[border:-border, border:-border]*255).astype(np.uint8))
    # close the writer to ensure that mp4 creation has the complete data available
    writer.close()
    temporalList.tofile("tempList.bin")
    VideoFileClip(outname+".gif").write_videofile(outname+".mp4")
    return






if __name__ == "__main__":
    
    args = parseArguments()
    parOpt = setupDevice(args)

    name = os.path.join("Clips",args.outname)
    
    ETH = args.ETH

    args.stacked = True
    data_set = setupDataset(args)    
    num_worker = 0 if (args.ETH) else 8
    loader = setupDataLoader(data_set, num_worker)
    

    sample_data = data_set[0]
    data_dims = sample_data[0].shape


    # Data information
    in_channels = data_dims[0]-3*9
    levels = data_dims[0]
    latRes = data_dims[1]
    lonRes = data_dims[2]
    args.in_channels = in_channels
    out_channels = args.classes
    if(args.binary):
        out_channels = 1
    

    #Print info
    if(parOpt.myRank == 0):
        print()
        print("Data...")
        print("Data-Location:", data_set.data_dir)
        print("Label-Location:", data_set.label_dir)
        print()
        print("Datalayout...")
        print("Resolution in (after crop):", data_dims[-2], data_dims[-1])
        print("Resolution out (after crop):", latRes, lonRes)
        print("Channels:", in_channels)
        print("levels:", levels)
        print("Labeltypes:", out_channels)
        print("")

    model = None
    if(not(args.ETH or args.preCalc)):
       model = setupModel(args, parOpt)

    num_samples = len(loader)
    if(args.num_samples != -1):
        num_samples = args.num_samples
    print("Evaluating {} Data files".format(num_samples))
    with torch.no_grad():
        performInference(model, loader, num_samples, parOpt, args)
