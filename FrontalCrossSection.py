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

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# Current Best
# Medium Bottle Net, 32 Batchsize, BottleneckLayer 128 256 128, 3 levels, lr = 0.01, lines +- 1
# ~ 45% validation loss 

from skimage import measure,morphology

from FrontPostProcessing import filterFronts
import netCDF4

from era5dataset.ERA5Reader.readNetCDF import equivalentPotentialTemp



class DistributedOptions():
    def __init__(self):
        self.myRank = -1
        self.device = -1
        self.local_rank = -1
        self.world_size = -1
        self.nproc_per_node = -1
        self.nnodes = -1
        self.node_rank = -1


def parseArguments():
    parser = argparse.ArgumentParser(description='FrontNet')
    parser.add_argument('--net', help='path no net')
    parser.add_argument('--data', help='path to folder containing data')
    parser.add_argument('--label', type = str, default = None, help='path to folder containing label')
    parser.add_argument('--outname', help='name of the output')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--device', type = int, default = 0, help = "number of device to use")
    parser.add_argument('--fullsize', action='store_true', help='test the network at the global scope')
    parser.add_argument('--NWS', action = 'store_true', help='use Resolution of hires')
    parser.add_argument('--num_samples', type = int, default = -1, help='number of samples to infere from the dataset')
    parser.add_argument('--classes', type = int, default = 1, help = 'How many classes the network should predict (binary case has 1 class denoted by probabilities)')
    parser.add_argument('--normType', type = int, default = 0, help = 'How to normalize the data: 0 min-max, 1 mean-var, 2/3 the same but per pixel')
    parser.add_argument('--labelGroupingList', type = str, default = None, help = 'Comma separated list of label groups \n possible fields are w c o s (warm, cold, occluson, stationary)')
    parser.add_argument('--ETH', action = 'store_true', help = 'Compare against an ETH result instead of net')
    parser.add_argument('--show-error', action = 'store_true', help = 'show the inividual error values during inference')
    parser.add_argument('--fromFile', type = str, default = None, help = 'show the inividual error values during inference')
    parser.add_argument('--calcType', type = str, default = "ML", help = 'from which fronts should the crossing be calculated')
    parser.add_argument('--calcVar', type = str, default = "t", help = 'which variable to measure along the cross section')
    args = parser.parse_args()
    args.binary = args.classes == 1
    
    return args

def setupDevice(args):
    parOpt = DistributedOptions()
    parOpt.myRank = 0
    if not args.disable_cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        parOpt.device = torch.device('cuda')
    else:
        parOpt.device = torch.device('cpu')
    return parOpt
    
def setupDataset(args):
    data_fold = args.data
    label_fold = args.label
    stepsize = 0.25
    if(args.fullsize):
        cropsize = (720, 1440)
        mapTypes = {"NA": ("NA", (90,-89.75), (-180,180), (-0.25,0.25)) }
        if(args.NWS):
            mapTypes = {"hires": ("hires", (90, -89.75), (-180, 180), (-0.25,0.25)) }
    else:
        cropsize = (184,360)
        mapTypes = {"NA": ("NA", (76,30.25), (-50,40), (-stepsize, stepsize), None)}
        if(args.NWS):
            cropsize = (184, 344) 
            mapTypes = {"hires": ("hires", (76, 30.25), (-141, -55), (-stepsize,stepsize), None) }
    # only atlantic 
    if(False):
        cropsize = (120,160)
        mapTypes = {"NA": ("NA", (60,30.25), (-50,-10), (-0.25,0.25))}
        if(args.NWS):
            mapTypes = {"hires": ("hires", (60, 30.25), (-50, -10), (-0.25,0.25)) }

    
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

    # append the basic wind to get the directions
    variables.insert(6,"base(u)")
    variables.insert(7,"base(v)")
    variables.insert(8,"ept")

    myEraExtractor = DerivativeFlippingAwareEraExtractor(variables, [], [], 0.0, 0 , 1, normType = normType, sharedObj = None)
    if(ETH):
        myEraExtractor = ETHEraExtractor()
    

    # Create Dataset
    data_set = WeatherFrontDataset(data_dir=data_fold, label_dir=label_fold, mapTypes = mapTypes, levelRange = myLevelRange, transform=myTransform, outSize=cropsize, labelThickness= labelThickness, label_extractor = myLabelExtractor, era_extractor = myEraExtractor, has_subfolds = (True, False), asCoords = False, removePrefix = 8)
    return data_set




    

def setupDataLoader(data_set, args):
    # Create DataLoader 
    indices = list(range(len(data_set)))
    sampler = SequentialSampler(data_set)

    loader = DataLoader(data_set, shuffle=False, 
    batch_size = 1, sampler = sampler, pin_memory = True, 
    collate_fn = collate_wrapper(args.stacked, False, 0), num_workers = 8)
    return loader

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def getValAlongNormal(image, var, udir, vdir, length, border, grad, orientation):
    directional = False
    avgVar = np.zeros((2*length+1, image.shape[2]))
    sqavgVar = np.zeros((2*length+1, image.shape[2]))
    avgVarBuf = np.zeros((2*length+1))
    sqavgVarBuf = np.zeros((2*length+1))
    numPoints = np.zeros((image.shape[2]))
    blength = max(border,length)
    for channel in range(image.shape[2]):
        channelImage = image[:,:,channel]
        if(directional):
            dirx = ndimage.sobel(channelImage, axis = 1)
            diry = ndimage.sobel(channelImage, axis = 0)
            dirx = np.roll(dirx, 1, axis=1)
            diry = np.roll(diry, 1, axis=0)

            grads = np.array([dirx,diry])
            dirx, diry = grads / (np.linalg.norm(grads, axis=0)+0.000001)
            angle = np.angle(dirx+1j*diry)
        wind = np.array([udir,vdir])
        udir, vdir = wind / (np.linalg.norm(wind, axis=0)+0.0000001)
        anglewind = np.angle(udir+1j*vdir)

        validPoints = np.nonzero(channelImage[blength:-blength,blength:-blength])
        localNumPoints = validPoints[0].shape[0]
        numPoints[channel] += localNumPoints
        
        for ppair in range(validPoints[0].shape[0]):
            py, px = blength+validPoints[0][ppair], blength+validPoints[1][ppair]
            negRang = 9
            posRang = 9+1
            myRegion = channelImage[py-negRang:py+posRang,px-negRang:px+posRang]
            if(len(np.nonzero(myRegion)[0])< 3):
                continue
            myNeighborhood = np.zeros(2)
            lab = measure.label(myRegion>0.5)
            tgtlab = lab[negRang,negRang]
            ori = measure.regionprops(lab)[tgtlab-1].orientation
            for mx in range(-negRang,posRang):
                for my in range(-negRang,posRang):
                    myVal = myRegion[my+negRang,mx+negRang]
                    myDist = np.array([mx,my])
                    if(abs(ori) > np.pi/4 and abs(ori) < 3*np.pi/4):
                        if(mx < 0):
                            myDist *= -1
                    else:
                        if(my < 0):
                            myDist *= -1
                    if my != 0 or mx != 0:
                        myNeighborhood += myVal*myDist/np.linalg.norm(myDist)
            if(np.linalg.norm(myNeighborhood) == 0):
                print(myRegion)
                print(ori)
                for mx in range(-negRang,posRang):
                    for my in range(-negRang,posRang):
                        myVal = myRegion[my+negRang,mx+negRang]
                        myDist = np.array([mx,my])
                        if(abs(ori) > np.pi/4 and abs(ori) < 3*np.pi/4):
                            if(mx < 0):
                                myDist *= -1
                        else:
                            if(my < 0):
                                myDist *= -1
                        if(myVal>0):
                            print(myDist)
                #ignore the point and continue with the next
                numPoints[channel] -= 1
                continue

            myNeighborhood /= np.linalg.norm(myNeighborhood)
            myYdir = myNeighborhood[0]
            myXdir = myNeighborhood[1]
            
            # normalize direction
            myLen = np.sqrt(myYdir*myYdir+myXdir*myXdir)
            myXdir /= myLen
            myYdir /= myLen
            
            
            pointsY = py-myYdir*np.arange(-length,length+1)
            pointsX = px+myXdir*np.arange(-length,length+1)
            # get the mean wind along the normal
            direction = [np.mean(bilinear_interpolate(udir, pointsX, pointsY)), np.mean(bilinear_interpolate(vdir, pointsX,pointsY))]
            # get the dot product of mean wind along the normal and the normal, to determine whether or not both are in the same direction
            direction = direction[0]*myXdir+direction[1]*myYdir
            if(direction <= 0):
                pointsX = np.flip(pointsX)
                pointsY = np.flip(pointsY)
            if(directional):
                windAngleBuf = np.abs(np.cos(bilinear_interpolate(anglewind, pointsX, pointsY)-angle[py,px]))
            else:
                windAngleBuf = 1
            avgVarBuf = bilinear_interpolate(var, pointsX, pointsY)*windAngleBuf
            sqavgVarBuf = bilinear_interpolate(var, pointsX, pointsY)*windAngleBuf
            tgtChannel = channel
            if(grad and orientation):
                for x in range(len(avgVarBuf)-1):
                    if(avgVarBuf[x+1]>2 and avgVarBuf[x] < -2):
                        avgVarBuf[x+1]-=2*np.pi
                    elif(avgVarBuf[x+1]<-2 and avgVarBuf[x] > 2):
                        avgVarBuf[x+1]+=2*np.pi

                avgVar[:,tgtChannel] += np.abs(np.cumsum(np.gradient(avgVarBuf)))
                sqavgVar[:,tgtChannel] += np.gradient(sqavgVarBuf)**2
            elif(grad and not orientation):
                avgVar[:,tgtChannel] += np.gradient(avgVarBuf)
                sqavgVar[:,tgtChannel] += np.gradient(sqavgVarBuf)**2
            else:
                avgVar[:,tgtChannel] += avgVarBuf
                sqavgVar[:,tgtChannel] += sqavgVarBuf**2
    return avgVar, sqavgVar, numPoints


def inferResults(model, inputs, args):
    if(args.ETH or args.calcType == "WS"):
        outputs = inputs.permute(0,2,3,1)
        smoutputs = inputs.permute(0,2,3,1)
    else:
        tgtIn = torch.cat((inputs[:,:6*9], inputs[:,-1:]), dim = 1)
        outputs = model(tgtIn)
        outputs = outputs.permute(0,2,3,1)
        smoutputs = torch.softmax(outputs.data, dim = -1)
        smoutputs[0,:,:,0] = 1-smoutputs[0,:,:,0]

        # If some labels are not to be considered additionally remove them from the 0 case (all others don't matter)
        labelsToUse = args.labelGroupingList.split(",")
        possLabels = ["w","c","o","s"]
        for idx, possLab in enumerate(possLabels, 1):
            isIn = False
            for labelGroup in labelsToUse:
                if(possLab in labelGroup):
                    isIn = True
            if(not isIn):
                smoutputs[0,:,:,0] -= smoutputs[0,:,:,idx]
        smoutputs = filterFronts(smoutputs.cpu().numpy(), 20)
    return outputs, smoutputs

def readSecondary(rootgrp, var, time, level, latrange, lonrange):
    vals = np.zeros((abs(int(latrange[0])-int(latrange[1]))*4, abs(int(lonrange[1])-int(lonrange[0]))*4))
    if(lonrange[0] < 0 and lonrange[1] >= 0):
        vals[:,:-int(lonrange[0])*4] =  rootgrp[var][time,level,int(90-latrange[0])*4:int(90-latrange[1])*4, int(lonrange[0])*4:]
        vals[:,-int(lonrange[0])*4:] = rootgrp[var][time,level,int(90-latrange[0])*4:int(90-latrange[1])*4, :int(lonrange[1])*4]
    else:
        vals[:,:] = rootgrp[var][time,level,int(90-latrange[0])*4:int(90-latrange[1])*4, int(lonrange[0])*4:int(lonrange[1])*4]
    return vals


def performInference(model, loader, num_samples, parOpt, args):
    length = 8
    out_channels = 4
    border = 20
    avgVar = np.zeros((2*length+1, out_channels))
    sqavgVar = np.zeros((2*length+1, out_channels))
    numPoints = np.zeros((out_channels))
    skip = 0#31*4+28*4+31*4+30*4+31*4+30*4+31*4
    for idx, data in enumerate(tqdm(loader, desc ='eval'), 0):
        if idx<skip:
            continue
        if(idx == num_samples+skip):
            break
        inputs, labels, filename = data
        inputs = inputs.to(device = parOpt.device, non_blocking=False)
        labels = labels.to(device = parOpt.device, non_blocking=False)
        # remove all short labels caused by cropping
        labels = filterFronts(labels.cpu().numpy(), border)
        _, outputs = inferResults(model, inputs, args)
        

        #Hard coded, for derivatives
        meanu, varu = 1.27024432, 6.74232481e+01
        meanv, varv = 1.0213897e-01, 4.36244384e+01
        meant, vart = 2.75355461e+02, 3.20404803e+02
        meanq, varq = 5.57926815e-03, 2.72627785e-05 
        meansp, varsp = 8.65211548e+04, 1.49460630e+08

        udir = inputs[0,9*6+8]
        vdir = inputs[0,9*7+8]
        
        # Generally no gradient (finite differences should be calculated)
        grad = False
        orientation = False

        data_set = loader.dataset
        no = data_set.removePrefix
        year,month,day,hour = filename[0][no:no+4],filename[0][no+4:no+6],filename[0][no+6:no+8],filename[0][no+9:no+11]
        # we do not have the 29th of february for ZFiles
        if("_z" in args.calcVar and month == "02" and day == "29"):
            continue
        mapType = "hires" if args.NWS else "NA"

        # determine variable, which should be evaluated
        if(args.calcVar == "t" or args.calcVar == "dt"):
            grad = args.calcVar == "dt"
            var = inputs[0,8]*np.sqrt(vart)+meant
        elif(args.calcVar == "q" or args.calcVar == "dq"):
            grad = args.calcVar == "dq"
            var = inputs[0,17]*np.sqrt(varq)+meanq
        elif(args.calcVar == "sp"):
            var = inputs[0,5*9+8]*np.sqrt(varsp)+meansp
        elif(args.calcVar == "wind"):
            # wind speed
            var = torch.abs(udir+1j*vdir)*2
        elif(args.calcVar == "wind_z"):
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/Z{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrange = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            u = readSecondary(rootgrp, "var131", 0, 9, tgt_latrange, tgt_lonrange)
            v = readSecondary(rootgrp, "var132", 0, 9, tgt_latrange, tgt_lonrange)
            var = torch.abs(torch.from_numpy(u+1j*v))
        elif(args.calcVar == "ept_z" or args.calcVar == "dept_z"):
            grad = args.calcVar == "dept_z"
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/Z{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrange = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            t = readSecondary(rootgrp, "var130", 0, 9, tgt_latrange, tgt_lonrange)
            q = readSecondary(rootgrp, "var133", 0, 9, tgt_latrange, tgt_lonrange)
            ept = equivalentPotentialTemp(t,q,85000)
            var = torch.from_numpy(ept)
            rootgrp.close()
        elif(args.calcVar == "t_z" or args.calcVar == "dt_z"):
            grad = args.calcVar == "dt_z"
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/Z{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrange = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            #t = rootgrp["var130"][0,9,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            t = readSecondary(rootgrp, "var130", 0, 9, tgt_latrange, tgt_lonrange)
            var = torch.from_numpy(t)
            print(var.shape)
            rootgrp.close()
        elif(args.calcVar == "q_z" or args.calcVar == "dq_z"):
            grad = args.calcVar == "dq_z"
            var = inputs[0,9*8+8]
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/Z{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrange = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            #q = rootgrp["var133"][0,9,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            q = readSecondary(rootgrp, "var133", 0, 9, tgt_latrange, tgt_lonrange)
            var = torch.from_numpy(q)
            rootgrp.close()
        elif(args.calcVar == "winddir"):
            # wind speed
            grad = True
            orientation = True
            var = torch.angle(udir+1j*vdir)
        elif(args.calcVar == "10mwinddir"):
            grad = True
            orientation = True
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/B{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            u10dir = rootgrp["u10"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            v10dir = rootgrp["v10"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            wind = torch.from_numpy(u10dir+1j*v10dir)
            var = torch.angle(wind)
            rootgrp.close()
        elif(args.calcVar == "10mwind"):
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/B{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            u10dir = rootgrp["u10"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            v10dir = rootgrp["v10"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            wind = torch.from_numpy(u10dir+1j*v10dir)
            var = torch.abs(wind)
            rootgrp.close()
        elif(args.calcVar == "precip"):
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/precip{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            print(tgt_lonrange, tgt_latrange)
            prec = rootgrp["tp"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            var = torch.from_numpy(prec)
        
        # Which kind of fronts should be tested (ML -> Network, WS -> WeatherService, OP -> over predicted (false positives), CP -> correct predicted (true positives, network oriented), 
        # NP -> not ptedicted (false negatives), CL correctly labeled (true positives, weather service oriented)
        if(args.calcType == "ML"):
            frontImage = outputs[0,:,:,1:]
        elif(args.calcType == "WS"):
            frontImage = labels[0,:,:,:]
        elif(args.calcType == "OP" or args.calcType == "CP"):
            # OverPrediction: All Predictions that are more than 3 pixel from the next GT Label
            # CorretPrediction: All Predictions that are no more than 3 pixel from the next GT Label
            frontImage = outputs[0,:,:,1:]
            for channel in range(labels.shape[-1]):
                if(args.calcType == "OP"):
                    distImg = distance_transform_edt(1-labels[0,:,:,channel], return_distances = True, return_indices = False) > 3
                    frontImage[:,:,channel] = frontImage[:,:,channel]*distImg
                elif(args.calcType == "CP"):
                    distImg = distance_transform_edt(1-labels[0,:,:,channel], return_distances = True, return_indices = False) <= 3
                    frontImage[:,:,channel] = frontImage[:,:,channel]*distImg
        
        elif(args.calcType == "NP" or args.calcType == "CL"):
            # NoPrediction: All Label that are more than 3 pixel from the next prediction
            # CorrectLabel: All Label that are are no more than 3 pixel from the next prediction
            outputs = outputs[0,:,:,1:]
            frontImage = labels[0]

            for channel in range(labels.shape[-1]):
                if(args.calcType == "NP"):
                    distImg = distance_transform_edt(1-(outputs[:,:,channel]), return_distances = True, return_indices = False) > 3
                    frontImage[:,:,channel] = frontImage[:,:,channel]*distImg
                elif(args.calcType == "CL"):
                    distImg = distance_transform_edt(1-(outputs[:,:,channel]), return_distances = True, return_indices = False) <= 3
                    frontImage[:,:,channel] = frontImage[:,:,channel]*distImg
        
        curravg, currsqavg, currnumPoints = getValAlongNormal(frontImage, var.cpu().numpy(), udir.cpu().numpy(), vdir.cpu().numpy(), length, border, grad, orientation)
        avgVar += curravg
        sqavgVar += currsqavg
        numPoints += currnumPoints

    means = avgVar / numPoints
    variances = sqavgVar / numPoints - means*means
    print(numPoints)
    return means, variances






if __name__ == "__main__":
    
    args = parseArguments()
    parOpt = setupDevice(args)

    name = os.path.join("CrossSections",args.outname)
    
    ETH = args.ETH

    args.stacked = True
    data_set = setupDataset(args)    
    loader = setupDataLoader(data_set, args)
    

    sample_data = data_set[0]
    data_dims = sample_data[0].shape


    # Data information
    in_channels = data_dims[0]-3*9
    levels = data_dims[0]
    latRes = data_dims[1]
    lonRes = data_dims[2]
    
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

    if(not ETH):
        #Load Net
        embeddingFactor = 6
        SubBlocks = (3,3,3)
        kernel_size = 5
        model = FDU2DNetLargeEmbedCombineModular(in_channel = in_channels, out_channel = out_channels, kernel_size = kernel_size, sub_blocks = SubBlocks, embedding_factor = embeddingFactor).to(parOpt.device)
        model.load_state_dict(torch.load(args.net, map_location = parOpt.device))
        model.eval()
        if(parOpt.myRank == 0):
            print()
            print("Begin Evaluation of Data...")
            print("Network:", model)

    num_samples = len(loader)
    if(args.num_samples != -1):
        num_samples = args.num_samples
    print("Evaluating {} Data files".format(num_samples))
    with torch.no_grad():
        avg_error, var_error = performInference(model, loader, num_samples, parOpt, args)

    # output 
    if(not os.path.isdir(name)):
        os.mkdir(name)
    region = "NWS" if args.NWS else "DWD"
    genName = os.path.join(name,region+"_"+args.calcType+"_crossSection_"+args.calcVar+"_")
    meanName = genName+"mean"
    varName = genName+"var"
    avg_error.tofile(meanName+".bin")
    var_error.tofile(varName+".bin")
    plt.plot(np.arange(-8,9), avg_error )
    plt.legend(["warm","cold","occ","stnry"])
    plt.savefig(meanName+".png")
    plt.clf()
    plt.plot(np.arange(-8,9), var_error)
    plt.legend(["warm","cold","occ","stnry"])
    plt.savefig(varName+".png")
        
