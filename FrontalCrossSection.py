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

from skimage import measure
from InferOutputs import inferResults, setupDataLoader, setupDevice, setupModel


import netCDF4

from metpy.calc import equivalent_potential_temperature, dewpoint_from_specific_humidity, relative_humidity_from_specific_humidity
from metpy.units import units
from geopy import distance

def rhi(p,q,T):
    eps     = 0.622 # Molmassenverhältnis von Wasser zu Luft.

    preal   = np.array(q * (p*100)/(eps + (1 - eps) * q)) # Dampfdruck in hPa 
               # in Abhängigkeit der spezifischen Feuchte(kg/kg) 
               # und des Drucks(Pa)
    picesat = np.array(np.exp(9.550426 - 5723.265/T + 3.53068 *
        np.log(T)-0.00728332 * T)) # Saettigungsdampfdruck ueber Eis

    rhi_return = preal/picesat
    return(rhi_return)

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
    parser.add_argument('--show-error', action = 'store_true', help = 'show the inividual error values during inference')
    parser.add_argument('--fromFile', type = str, default = None, help = 'show the inividual error values during inference')
    parser.add_argument('--calcType', type = str, default = "ML", help = 'from which fronts should the crossing be calculated')
    parser.add_argument('--calcVar', type = str, default = "t", help = 'which variable to measure along the cross section')
    parser.add_argument('--secPath', type = str, default = None, help = 'Path to folder with secondary data containing variable information to be evaluated. Data should be stored as <secPath>/YYYY/MM/<fileID>YYYYMMDD_HH.nc . <fileID> is an Identifier based on the type of file (e.g. ml,B,Z,precip)')
    args = parser.parse_args()
    args.binary = args.classes == 1
    
    return args
    
def setupDataset(args):
    data_fold = args.data
    label_fold = args.label
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
    # adjusted for general use
    subfolds = (False, False)
    remPref = 0

    if(args.preCalc):
        myEraExtractor = BinaryResultExtractor()
        subfolds = (False, False)
        remPref = 0
    

    # Create Dataset
    data_set = WeatherFrontDataset(data_dir=data_fold, label_dir=label_fold, mapTypes = mapTypes, levelRange = myLevelRange, transform=myTransform, outSize=cropsize, labelThickness= labelThickness, label_extractor = myLabelExtractor, era_extractor = myEraExtractor, has_subfolds = subfolds, asCoords = False, removePrefix = remPref)
    return data_set


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

def getSamplePosition(py, px, ydir, xdir, length):
    myYpoints = py - np.arange(-length, length+1)*ydir
    myXpoints = px + np.arange(-length, length+1)*xdir
    return myYpoints,myXpoints

def getSamplePositionCirc(source, offset, ydir, xdir, length):

    dist = 20
    py = source[0]+offset[0]
    px = source[1]+offset[1]
    xdists = np.zeros(2*length+1)     
    ydists = np.zeros_like(xdists)
    pyd = (90-py*0.25)
    pxd = -180 + px*0.25

    for l in range(-length, length+1,1):
        dest = distance.distance(kilometers=l*dist).destination((pyd, pxd), np.angle(ydir+1j*xdir)*180/np.pi)
        # get pixel posiitions 
        xdists[l+length] = (dest[1]+180)*4-offset[1]
        ydists[l+length] = (90-dest[0])*4-offset[0]
    return ydists, xdists



def getValAlongNormal(image, var, udir, vdir, length, border, grad, orientation, offset):
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
                    # we do not flip the sign, as we implicitly convert the y axis into a northward axis from southward
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
            # myYdir is northward
            myYdir = myNeighborhood[0]
            # myXdir is eastward
            myXdir = myNeighborhood[1]
            
            # normalize direction
            myLen = np.sqrt(myYdir*myYdir+myXdir*myXdir)
            myXdir /= myLen
            myYdir /= myLen
            
            # get a better estimation respecting the projection 
            pointsY, pointsX = getSamplePositionCirc((py, px), offset, myYdir, myXdir, length)
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
                # grad 2 and 3 are for the TFP
                if(grad == 1):
                    avgVar[:,tgtChannel] += np.gradient(avgVarBuf)
                    sqavgVar[:,tgtChannel] += np.gradient(sqavgVarBuf)**2
                elif(grad==2):
                    print("We use this here!", grad)
                    avgVar[:,tgtChannel] += np.gradient(np.gradient(avgVarBuf))
                    sqavgVar[:,tgtChannel] += np.gradient(np.gradient(sqavgVarBuf))**2
                elif(grad==3):
                    print("We use this here!", grad)
                    avgVar[:,tgtChannel] += np.gradient(np.gradient(np.gradient(avgVarBuf)))
                    sqavgVar[:,tgtChannel] += np.gradient(np.gradient(np.gradient(sqavgVarBuf)))**2
            else:
                avgVar[:,tgtChannel] += avgVarBuf
                sqavgVar[:,tgtChannel] += sqavgVarBuf**2
    return avgVar, sqavgVar, numPoints


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


def getWindData(file, calcVar, latrange, lonrange):
    rootgrpwind = netCDF4.Dataset(os.path.realpath(file), "r", format="NETCDF4", parallel=False)
    if("_z" in calcVar):
        udir = readSecondary(rootgrpwind, "var131", 0, 9, latrange, lonrange)
        vdir = readSecondary(rootgrpwind, "var132", 0, 9, latrange, lonrange)
    elif("_b" in calcVar or "_precip" in calcVar):
        udir = readSecondary(rootgrpwind, "u10", 0, None, latrange, lonrange)
        vdir = readSecondary(rootgrpwind, "v10", 0, None, latrange, lonrange)
    elif("_ml" in calcVar):
        udir = readSecondary(rootgrpwind, "u", 0, -1, latrange, (lonrange[0]+180,lonrange[1]+180))
        vdir = readSecondary(rootgrpwind, "v", 0, -1, latrange, (lonrange[0]+180,lonrange[1]+180))
    rootgrpwind.close()
    return udir, vdir


def getSecondaryData(file, calcVar, latrange, lonrange):
    rootgrp = netCDF4.Dataset(os.path.realpath(file), "r", format="NETCDF4", parallel=False)
    # determine variable, which should be evaluated at lowest model level (L137)
    # ml files are [-180 to 180]E => add the 180 offset when reading from those files
    # TODO-- update to the more general read method from the reader
    if(calcVar == "t_ml" or calcVar == "dt_ml"):
        grad = calcVar == "dt_ml"
        var = readSecondary(rootgrp, "t", 0, -1, latrange, (lonrange[0]+180,lonrange[1]+180))
    elif(calcVar == "q_ml" or calcVar == "dq_ml"):
        grad = calcVar == "dq_ml"
        var = readSecondary(rootgrp, "q", 0, -1, latrange, (lonrange[0]+180,lonrange[1]+180))
    elif(calcVar == "wind_ml"):
        # wind speed
        u = readSecondary(rootgrp, "u", 0, -1, latrange, (lonrange[0]+180,lonrange[1]+180))
        v = readSecondary(rootgrp, "v", 0, -1, latrange, (lonrange[0]+180,lonrange[1]+180))
        var = torch.abs(torch.from_numpy(u+1j*v)).cpu().numpy()
    elif(calcVar == "winddir_ml"):
        # wind speed
        grad = True
        orientation = True
        u = readSecondary(rootgrp, "u", 0, -1, latrange, (lonrange[0]+180,lonrange[1]+180))
        v = readSecondary(rootgrp, "v", 0, -1, latrange, (lonrange[0]+180,lonrange[1]+180))
        var = torch.angle(torch.from_numpy(u+1j*v)).cpu().numpy()
    elif(calcVar == "cc_ml"):
        # wind speed
        var = readSecondary(rootgrp, "cc", 0, -23, latrange, (lonrange[0]+180,lonrange[1]+180))

    # Read 850 hPA instead
    elif(calcVar == "wind_z"):
        u = readSecondary(rootgrp, "var131", 0, 9, latrange, lonrange)
        v = readSecondary(rootgrp, "var132", 0, 9, latrange, lonrange)
        var = torch.abs(torch.from_numpy(u+1j*v)).cpu().numpy()
    elif(calcVar == "ept_z" or calcVar == "dept_z" or calcVar == "tfp_z" or calcVar == "dtfp_z"):
        t = units.Quantity(readSecondary(rootgrp, "var130", 0, 9, latrange, lonrange), "K")
        q = readSecondary(rootgrp, "var133", 0, 9, latrange, lonrange)
        p = units.Quantity(85000, "Pa")
        dewp = dewpoint_from_specific_humidity(p,t,q)
        ept = equivalent_potential_temperature(p,t,dewp)
        var = ept.magnitude
    elif(calcVar == "t_z" or calcVar == "dt_z"):
        t = readSecondary(rootgrp, "var130", 0, 9, latrange, lonrange)
        var = t
        print(var.shape)
    elif(calcVar == "q_z" or calcVar == "dq_z"):
        q = readSecondary(rootgrp, "var133", 0, 9, latrange, lonrange)
        var = q
    elif(calcVar == "rq_z" or calcVar == "drq_z"):
        t = units.Quantity(readSecondary(rootgrp, "var130", 0, 9, latrange, lonrange), "K")
        q = readSecondary(rootgrp, "var133", 0, 9, latrange, lonrange)
        p = units.Quantity(85000, "Pa")
        rq = relative_humidity_from_specific_humidity(p,t,q)
        var = rq.magnitude
    elif(calcVar == "oversat_z"):
        t = units.Quantity(readSecondary(rootgrp, "var130", 0, 9, latrange, lonrange), "K")
        q = readSecondary(rootgrp, "var133", 0, 9, latrange, lonrange)
        p = units.Quantity(85000, "Pa")
        rq = relative_humidity_from_specific_humidity(p,t,q)
        var = (rq.magnitude>1.0)*1.0
    elif(calcVar == "rqi_z" or calcVar == "drqi_z"):
        t = units.Quantity(readSecondary(rootgrp, "var130", 0, 3, latrange, lonrange), "K")
        q = readSecondary(rootgrp, "var133", 0, 3, latrange, lonrange)
        p = units.Quantity(30000, "Pa")
        rq = rhi(p.magnitude/100,q, t.magnitude)
        var = (rq * (t.magnitude>200) * (t.magnitude < 233))*1.0
    elif(calcVar == "iceoversat_z"):
        t = units.Quantity(readSecondary(rootgrp, "var130", 0, 3, latrange, lonrange), "K")
        q = readSecondary(rootgrp, "var133", 0, 3, latrange, lonrange)
        p = units.Quantity(30000, "Pa")
        rq = rhi(p.magnitude/100,q, t.magnitude)
        var = ((rq>1.0) * (t.magnitude>200) * (t.magnitude < 233))*1.0
    # These should work, but are not tested!
    # 10m winds
    elif(calcVar == "winddir_b"):
        u10dir = readSecondary(rootgrp, "u10", 0, None, latrange, lonrange)
        v10dir = readSecondary(rootgrp, "v10", 0, None, latrange, lonrange)
        wind = torch.from_numpy(u10dir+1j*v10dir)
        var = torch.angle(wind).cpu().numpy()
    elif(calcVar == "wind_b"):
        u10dir = readSecondary(rootgrp, "u10", 0, None, latrange, lonrange)
        v10dir = readSecondary(rootgrp, "v10", 0, None, latrange, lonrange)
        wind = torch.from_numpy(u10dir+1j*v10dir)
        var = torch.abs(wind).cpu().numpy()
    elif(calcVar == "sp_b"):
        var = readSecondary(rootgrp, "sp", 0, None, latrange, lonrange)
    elif(calcVar == "lcc_b"):
        var = readSecondary(rootgrp, "lcc", 0, None, latrange, lonrange)
    elif(calcVar == "mcc_b"):
        var = readSecondary(rootgrp, "mcc", 0, None, latrange, lonrange)
    elif(calcVar == "hcc_b"):
        var = readSecondary(rootgrp, "hcc", 0, None, latrange, lonrange)
    # precipitation
    elif(calcVar == "tp_precip"):
        prec = readSecondary(rootgrp, "tp", 0, None, latrange, lonrange)
        var = prec
    elif(calcVar == "extremetp_precip"):
        prec = readSecondary(rootgrp, "tp", 0, None, latrange, lonrange)
        file2 = "tpThresh.nc"
        rootgrp2 = netCDF4.Dataset(os.path.realpath(file2), "r", format="NETCDF4", parallel=False)
        precoff = readSecondary(rootgrp2, "tp", 0, None, latrange, lonrange)
        rootgrp2.close()
        var = (prec>precoff)*1.0
    rootgrp.close()
    return var

def getModifier(calcVar):
    orientation = "winddir" in calcVar
    grad = (calcVar[0]=="d" or orientation)*1
    if("tfp" in calcVar):
        grad = 2
    if("dtfp" in calcVar):
        grad = 3
    return grad, orientation

def getDate(filename, no):
    year,month,day,hour = filename[no:no+4],filename[no+4:no+6],filename[no+6:no+8],filename[no+9:no+11]
    return year,month,day,hour

def getTgtRange(data_set, mapType):
    latoff= (data_set.mapTypes[mapType][1][0]-90)/data_set.mapTypes[mapType][3][0]
    lonoff= (data_set.mapTypes[mapType][2][0]+180)/data_set.mapTypes[mapType][3][1]
    print("offsets for the corresponding mapType, to estimate distance in km:", latoff, lonoff)
    tgt_latrange, tgt_lonrange = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
    return tgt_latrange, tgt_lonrange


def getSecondaryFile(calcVar, path, year, month, day, hour):
    if("_z" in calcVar):
        newFile = os.path.join(path, year, month,"Z{0}{1}{2}_{3}.nc".format(year,month,day,hour))
    if("_b" in calcVar):
        newFile = os.path.join(path, year, month,"B{0}{1}{2}_{3}.nc".format(year,month,day,hour))
    if("_precip" in calcVar):
        newFile = os.path.join(path, year, month,"precip{0}{1}{2}_{3}.nc".format(year,month,day,hour))
    if("_ml" in calcVar):
        newFile = os.path.join(path, year, month,"ml{0}{1}{2}_{3}.nc".format(year,month,day,hour))
    return newFile

def getWindFile(calcVar, path, year, month, day, hour):
    if("_z" in calcVar):
        windFile = os.path.join(path, year, month,"Z{0}{1}{2}_{3}.nc".format(year,month,day,hour))
    if("_b" in calcVar):
        windFile = os.path.join(path, year, month,"B{0}{1}{2}_{3}.nc".format(year,month,day,hour))
    if("_precip" in calcVar):
        windFile = os.path.join(path, year, month,"B{0}{1}{2}_{3}.nc".format(year,month,day,hour))
    if("_ml" in calcVar):
        windFile = os.path.join(path, year, month,"ml{0}{1}{2}_{3}.nc".format(year,month,day,hour))
    return windFile


def performInference(model, loader, num_samples, parOpt, args):
    
    # 10 pixel in each direction a 20 km = 200km before and after front are checked
    length = 10
    out_channels = 4
    # border has a size of 20 pixel due to the network and an additional 20 pixel buffer for the sampling along the normal
    border = 40
    avgVar = np.zeros((2*length+1, out_channels))
    sqavgVar = np.zeros((2*length+1, out_channels))
    numPoints = np.zeros((out_channels))
    skip = 0#31*4+28*4+31*4+30*4+31*4+30*4+31*4

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
            labels = labels.to(device = parOpt.device, non_blocking=False)
        
        # remove all short labels (# 1 is a dummy value, evaluation will skip 40 pixel anyways)
        labels = filterFronts(labels.cpu().numpy(), 1)
        if(args.calcType == "WS"):
            outputs = inputs.permute(0,2,3,1)
        elif(args.preCalc):
            outputs = (inputs*1).cpu().numpy()
            print(outputs.shape)
        else:
            args.border = 5
            outputs = inferResults(model, inputs, args).cpu().numpy()
                
        year,month,day,hour = getDate(filename[0], no)

        # we do not have the 29th of february for ZFiles
        if("_z" in args.calcVar and month == "02" and day == "29"):
            continue
        
        newFile = getSecondaryFile(args.calcVar, secondaryPath, year, month, day, hour)
        windFile = getWindFile(args.calcVar, secondaryPath, year, month, day, hour)
        
        udir, vdir = getWindData(windFile, args.calcVar, tgt_latrange, tgt_lonrange)
        # Generally no gradient (finite differences should be calculated)
        var = getSecondaryData(newFile, args.calcVar, tgt_latrange, tgt_lonrange)
        grad, orientation = getModifier(args.calcVar)
        
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
        
        curravg, currsqavg, currnumPoints = getValAlongNormal(frontImage, var, udir, vdir, length, border, grad, orientation, (latoff,lonoff))
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

    args.stacked = True
    data_set = setupDataset(args)    
    num_worker = 0 if (args.calcType == "WS") else 8
    loader = setupDataLoader(data_set, num_worker)
    

    sample_data = data_set[0]
    data_dims = sample_data[0].shape


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
    if(not(args.calcType == "WS" or args.preCalc)):
       model = setupModel(args, parOpt)

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
    plt.plot(np.arange(-10,11), avg_error )
    plt.legend(["warm","cold","occ","stnry"])
    plt.savefig(meanName+".png")
    plt.clf()
    plt.plot(np.arange(-10,11), var_error)
    plt.legend(["warm","cold","occ","stnry"])
    plt.savefig(varName+".png")
        
