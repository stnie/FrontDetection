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
from skimage.io import imsave
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
    parser.add_argument('--label', type = str, default = None, help='path to folder containing label')
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
    parser.add_argument('--make_img', default = False, action='store_true', help='Create a single image instead')
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
        # add 5 degree to the input, such that we can savely extract lines from fronts at the corner of evaluation
        cropsize = (45*4,90*4)
        mapTypes = {"NA": ("NA", (75,30.25), (-50,40), (-stepsize, stepsize), None)}
        if(args.NWS):
            cropsize = (45*4, 85*4) 
            mapTypes = {"hires": ("hires", (75, 30.25), (-140, -55), (-stepsize,stepsize), None) }
    else:
        # add 5 degree to the input, such that we can savely extract lines from fronts at the corner of evaluation
        cropsize = (46*4,90*4)
        mapTypes = {"NA": ("NA", (76,30.25), (-50,40), (-stepsize, stepsize), None)}
        if(args.NWS):
            cropsize = (46*4, 86*4) 
            mapTypes = {"hires": ("hires", (76, 30.25), (-141, -55), (-stepsize,stepsize), None) }
    
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

    subfolds = (False, False)
    remPref = 0

    if(ETH):
        myEraExtractor = ETHEraExtractor()
        subfolds = (False, False)
        remPref = 1
    if(args.preCalc):
        myEraExtractor = BinaryResultExtractor()
        subfolds = (False, False)
        remPref = 0
    

    # Create Dataset
    data_set = WeatherFrontDataset(data_dir=data_fold, label_dir = args.label, mapTypes = mapTypes, levelRange = myLevelRange, transform=myTransform, outSize=cropsize, labelThickness= labelThickness, label_extractor = myLabelExtractor, era_extractor = myEraExtractor, has_subfolds = subfolds, asCoords = False, removePrefix = remPref)
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


def CreateImageWithBackground(data, variableBg, surfaceBg, calcVar, alpha, rgb, off=1):
    #switch channels for usual output colors
    outpred = np.zeros((data.shape[0],data.shape[1],3))
    # red -> warm
    outpred[:,:,0] = data[:,:,off]
    # green -> stationary
    outpred[:,:,1] = data[:,:,3+off]
    # blue -> cold
    outpred[:,:,2] = data[:,:,1+off]
    # pink -> occlusion
    outpred[:,:,0] = (outpred[:,:,0]<=data[:,:,2+off])*data[:,:,2+off] + (outpred[:,:,0] > data[:,:,2+off])*outpred[:,:,0]

    outpred[:,:,2] = (outpred[:,:,2]<=data[:,:,2+off])*data[:,:,2+off] + (outpred[:,:,2] > data[:,:,2+off])*outpred[:,:,2]
    # white -> no clear distinction
    # get all Zeros
    zeros = np.nonzero((np.sum(outpred, axis=-1)==0)*1.0)
    # at all zero positions write a yellow line if the general Front is identified
    outpred[zeros[0],zeros[1],0] = data[zeros[0],zeros[1],0]
    outpred[zeros[0],zeros[1],1] = data[zeros[0],zeros[1],0]
    #outpred[zeros[0],zeros[1],2] = outputs[0,zeros[0],zeros[1],0]
    
    # normalize it to 0..1
    mini, maxi = getValueRanges(calcVar)
    variableBg = (variableBg - mini) / (maxi-mini)
    mygifImg = outpred
    sumimg = np.sum(mygifImg, axis = -1) < 0.5
    # if a front is present => print the front, else print the background variable
    mygifImg[:,:,0] = (alpha*surfaceBg + (1-alpha) * variableBg * (rgb[0]/255.0)) * sumimg + mygifImg[:,:,0] * (~sumimg) 
    mygifImg[:,:,1] = (alpha*surfaceBg + (1-alpha) * variableBg * (rgb[1]/255.0)) * sumimg + mygifImg[:,:,1] * (~sumimg)
    mygifImg[:,:,2] = (alpha*surfaceBg + (1-alpha) * variableBg * (rgb[2]/255.0)) * sumimg + mygifImg[:,:,2] * (~sumimg)
    return mygifImg


def performInference(model, loader, num_samples, parOpt, args):
    
    outfold = os.path.join("Clips", args.outname)
    if not os.path.isdir(outfold):
        os.mkdir(outfold)
    outname = os.path.join(outfold, "clip")
    border = 20
    months = [0,31, 29, 31,30, 31,30,31,31,30,31,30,31]
    # offset in days until start of month
    cumMonths = np.cumsum(months)
    skip = 0
    if(args.make_img):
        # to get september for image output
        skip = 4*(cumMonths[8]+13)-1
        if args.NWS:
            skip = 8*(cumMonths[8]+13)

    # Path to Secondary File
    secondaryPath = args.secPath
    if(secondaryPath is None):
        print("Secondary Path needed for this type of evaluation!")
        exit(1)
    
    data_set = loader.dataset
    no = data_set.removePrefix
    # Get Range
    mapType = "hires" if args.NWS else "NA"
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
        print("Continue without background image")
    if(not args.make_img):
        # If no image is to be made => a video should be made instead => Use gif writer 
        writer = imageio.get_writer(outname+".gif", mode="I", duration=0.1)
    for idx, data in enumerate(tqdm(loader, desc ='eval'), 0):
        if idx<skip:
            continue
        if(idx == num_samples+skip):
            break
        if(not torch.cuda.is_available()):
            inputs, labels, filename = data.data, data.labels, data.filenames
        else:
            inputs, labels, filename = data
            inputs = inputs.cpu().numpy()
        
        # remove all short labels (# 1 is a dummy value, evaluation will skip 40 pixel anyways)
        if(args.ETH):
            outputs = inputs.permute(0,2,3,1)
        elif(args.preCalc):
            outputs = (inputs*1).cpu().numpy()
            print(outputs.shape)
        else:
            outputs = inferResults(model, inputs, args)
        

        # Prepare the secondary data
        year,month,day,hour = getDate(filename[0], no)  

        # we do not have the 29th of february for ZFiles
        if("_z" in args.calcVar and month == "02" and day == "29"):
            continue
        
        newFile = getSecondaryFile(args.calcVar, secondaryPath, year, month, day, hour)
        var = getSecondaryData(newFile, args.calcVar, tgt_latrange, tgt_lonrange)
        
        # reshuffle input for display
        mygifImg = CreateImageWithBackground(outputs[0], var, contourMap, args.calcVar.split("_")[0], args.alpha, args.rgb)
        make_image=args.make_img
        if(make_image):
            mygifLab = CreateImageWithBackground(labels[0].cpu().numpy(), var, contourMap, args.calcVar.split("_")[0], args.alpha, args.rgb,0)
            # Create the diff img
            mydiffIn = np.zeros_like(labels[0].cpu().numpy())
            mydiffIn[:,:,0] = np.max(labels[0].cpu().numpy(), axis=-1)
            # Use Blue for Colorblindness
            mydiffIn[:,:,1] = np.max(outputs[0], axis=-1)
            # If detection and label overlap, show detection (to prevent color mixture)
            tmp = np.nonzero(mydiffIn[:,:,3])
            mydiffIn[tmp[0],tmp[1],0] = 0
            mygifDiff = CreateImageWithBackground(mydiffIn, var, contourMap, args.calcVar.split("_")[0], args.alpha, args.rgb,0)
            imsave(os.path.join(outfold,filename[0]+"diff.png"), (mygifDiff[border:-border, border:-border]*255).astype(np.uint8))
            imsave(os.path.join(outfold,filename[0]+"img.png"), (mygifImg[border:-border, border:-border]*255).astype(np.uint8))
            imsave(os.path.join(outfold,filename[0]+"lab.png"), (mygifLab[border:-border, border:-border]*255).astype(np.uint8))
        else:
            # add the image to the gif
            writer.append_data((mygifImg[border:-border, border:-border]*255).astype(np.uint8))
    if(not args.make_img):
        # close the writer to ensure that mp4 creation has the complete data available
        writer.close()
        # TODO -- safely remove the gif after mp4 creation.
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
    #print(data_dims)
    #imsave("tmp2.png", sample_data[0][0,:,:])


    # Data information
    in_channels = data_dims[0]
    levels = data_dims[0]
    latRes = data_dims[1]
    lonRes = data_dims[2]
    args.in_channels = in_channels
    out_channels = args.classes
    if(args.binary):
        out_channels = 1
    args.out_channels = out_channels
    

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
