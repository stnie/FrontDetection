import numpy as np
import torch
import os

from scipy.ndimage import distance_transform_edt

from Models.FDU3D import *

from tqdm import tqdm
import argparse

from era5dataset.FrontDataset import *
# ERA Extractors
from era5dataset.EraExtractors import *

from IOModules.csbReader import *

from NetInfoImport import *

import netCDF4
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm

from InferOutputs import setupDataLoader


def parseArguments():
    parser = argparse.ArgumentParser(description='FrontNet')
    parser.add_argument('--data', type = str, help = 'path to fronts either fold or file')
    parser.add_argument('--mask', type = str, help = 'path to mask netCDF4 file for extreme events')
    parser.add_argument('--heightMap', type=str, help = 'path to netCDF4 containing height map as geopotential')
    parser.add_argument("--season", type = str, default = "", help = "which season to evaluate for (djf, mam, jja, son), default no season")
    parser.add_argument('--calcVar', type = str, default = "precip", help = 'which variable to measure along the cross section')
    parser.add_argument('--num_samples', type = int, default = -1, help='number of samples to infere from the dataset')
    parser.add_argument('--outname', help='name of the output')
    '''
    parser.add_argument('--net', help='path no net')
    parser.add_argument('--data', help='path to folder containing data')
    parser.add_argument('--label', type = str, default = None, help='path to folder containing label')
    
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--device', type = int, default = 0, help = "number of device to use")
    parser.add_argument('--fullsize', action='store_true', help='test the network at the global scope')
    parser.add_argument('--NWS', action = 'store_true', help='use Resolution of hires')
    
    parser.add_argument('--classes', type = int, default = 1, help = 'How many classes the network should predict (binary case has 1 class denoted by probabilities)')
    parser.add_argument('--normType', type = int, default = 0, help = 'How to normalize the data: 0 min-max, 1 mean-var, 2/3 the same but per pixel')
    parser.add_argument('--labelGroupingList', type = str, default = None, help = 'Comma separated list of label groups \n possible fields are w c o s (warm, cold, occluson, stationary)')
    parser.add_argument('--ETH', action = 'store_true', help = 'Compare against an ETH result instead of net')
    parser.add_argument('--show-error', action = 'store_true', help = 'show the inividual error values during inference')
    parser.add_argument('--fromFile', type = str, default = None, help = 'show the inividual error values during inference')
    parser.add_argument('--calcType', type = str, default = "ML", help = 'from which fronts should the crossing be calculated')
    
    '''
    args = parser.parse_args()
    #args.binary = args.classes == 1
    
    return args

def setupDataset(args):
    data_fold = args.data
    cropsize = (720, 1440)
    mapTypes = {"all": ("", (90,-89.75), (-180,180), (-0.25,0.25)) }

    myLevelRange = np.arange(105,138,4)

    myTransform = (None, None)
    labelThickness = 1

    myEraExtractor = BinaryResultExtractor() 

    # Create Dataset
    data_set = WeatherFrontDataset(data_dir=data_fold, mapTypes = mapTypes, levelRange = myLevelRange, transform=myTransform, outSize=cropsize, labelThickness= labelThickness, era_extractor = myEraExtractor, has_subfolds = (False, False), asCoords = False, removePrefix = 0)
    return data_set


def performInference(loader, num_samples, parOpt, args):
    border = 20
    
    # number of iterations of dilation
    Boxsize = 10

    data_set = loader.dataset
    no = data_set.removePrefix
    mapType = "all"
    
    # get the day per month
    dpm = [0,31,29,31,30,31,30,31,31,30,31,30,31]
    # cumulative sum to get the number of hours since YYYY-12-01_00
    ssf = np.cumsum(np.array(dpm))*24
    seasons = ["djf", "mam", "jja", "son"]
    tgt_season = args.season

    tgtvar = ""
    if args.calcVar == "precip":
        front_file = args.data
        tgtvar = "tp"
        mask_file = args.mask
        # if it is a folder we will automatically look for an appropriate file
        if(os.path.isdir(args.mask)):
            mask_file = os.path.join(args.mask, "tmp2016_eventMask_{}.nc".format(tgt_season))
        height_file = args.heightMap
    else:
        print("Type not implemented. Abort!")
        exit(1)
    # data is a folder => use the dataset to read all data
    # else read a single file (that is already processed!)
    singleFiles =os.path.isdir(front_file)
    
    skip = 0
    total_fronts = np.zeros((num_samples, 680, 1400, 5), dtype=np.bool)
    if(singleFiles):
        # load all files individually and combine them within this script
        for idx, data in enumerate(tqdm(loader, desc ='eval'), 0):
            if idx<skip:
                continue
            if(idx == num_samples+skip):
                exit(1)
                break
            inputs, labels, filename = data.data.cpu().numpy(), data.labels, data.filenames
            inputs = inputs[:,border:-border,border:-border,:]

            front = inputs[0]
            for ftype in range(5):
                front[:,:,ftype] = distance_transform_edt(1-front[:,:,ftype])<=Boxsize
            total_fronts[idx,:,:,:] = front[:,:,:].astype(np.bool)
        # Uncomment to Write the calculated fronts as a single file 
        total_fronts.tofile(os.path.join(args.mask, "tmp2016_front4d_l2_v2.bin"))
        exit(1)
    else:
        # Already precalculated fronts in a single file. Load only once. Also No need for widening!
        total_fronts = np.fromfile(front_file, dtype=np.bool).reshape(-1,680,1400,5)

    if(tgt_season == "djf"):
        #the input file is ordered chronologically so d is the last entry and jf the first two months
        total_fronts = np.concatenate((total_fronts[:ssf[2]], total_fronts[ssf[11]:ssf[12]]), axis=0)
    elif(tgt_season == "mam"):
        total_fronts = total_fronts[ssf[2]:ssf[5]]
    elif(tgt_season == "jja"):
        total_fronts = total_fronts[ssf[5]:ssf[8]]
    elif(tgt_season == "son"):
        print(total_fronts.shape)
        total_fronts = total_fronts[ssf[8]:ssf[11]]
        print(total_fronts.shape)
    

    # Read the event mask
    rootgrp = netCDF4.Dataset(os.path.realpath(mask_file), "r", format="NETCDF4", parallel=False)
    tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
    # the files have lat 90 - -90,   lon 0 - 360
    # => we need to offset lonrange
    exEvs = np.zeros((rootgrp["time"][:].shape[0], abs(int(tgt_latrange[0])-int(tgt_latrange[1]))*4, abs(int(tgt_lonrage[1])-int(tgt_lonrage[0]))*4), dtype =np.bool)
    print(exEvs.shape[0], num_samples)
    if(tgt_lonrage[0] < 0 and tgt_lonrage[1] >= 0):
        exEvs[:,:,:-int(tgt_lonrage[0])*4] =  (rootgrp[tgtvar][:,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:]).astype(np.bool)
        exEvs[:,:,-int(tgt_lonrage[0])*4:] = (rootgrp[tgtvar][:,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, :int(tgt_lonrage[1])*4]).astype(np.bool)
    else:
        exEvs = rootgrp[tgtvar][:,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
    rootgrp.close()

    # our output has a border => so add this to the mask file, too
    exEvs = exEvs[:,border:-border,border:-border] 
    # create the aggregated events
    evcount = np.sum(exEvs, axis=0)



    # read the height map and remove all height +5 pixel radius
    rootgrp = netCDF4.Dataset(os.path.realpath(height_file), "r", format="NETCDF4", parallel=False)
    tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
    # the files have lat 90 - -90,   lon 0 - 360
    # => we need to offset lonrange
    heightmap = np.zeros((abs(int(tgt_latrange[0])-int(tgt_latrange[1]))*4, abs(int(tgt_lonrage[1])-int(tgt_lonrage[0]))*4))
    if(tgt_lonrage[0] < 0 and tgt_lonrage[1] >= 0):
        heightmap[:,:-int(tgt_lonrage[0])*4] =  (rootgrp["z"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:])
        heightmap[:,-int(tgt_lonrage[0])*4:] = (rootgrp["z"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, :int(tgt_lonrage[1])*4])
    else:
        heightmap = rootgrp["z"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
    rootgrp.close()


    # transform geopotential to height in m
    heightmap /= 9.80665
    # get all points below 2000m (such that point >2000m are background for distance transform)
    heightmap = heightmap[border:-border,border:-border] < 2000
    # Get all points that are more than 5 points away from any background (point > 2000m)
    heightmap = distance_transform_edt(heightmap) > 5
    # save map for future reference
    heightmap.tofile("heightFilterMap.bin")

    
    # initialize with set seed for reproducability
    rng = np.random.default_rng(12345)

    
    # when evaluating seasons, we should expect less extreme events per season
    casePerPoint = 13
    pointsPerList = 6
    # no specific season chosen => whole year. We now take ~4*13 ~ 50 cases per Point instead
    if(not (tgt_season in seasons)):
        casePerPoint = 50
    # fpop is the aggregate of the season
    fpop = np.sum(total_fronts, axis=0)/total_fronts.shape[0]
    # generate the event lists
    # get the total count of extreme events
    # get the ratio of front events 
    
    k = 0
    # use 1% steps
    stepsize = 1
    # Only use points within [20 : 60] N/S as basepoints
    fpop = np.concatenate((fpop[100:261],fpop[420:581]),axis=0)
    heightmap = np.concatenate((heightmap[100:261], heightmap[420:581]), axis=0)
    validHeights = np.nonzero(heightmap)
    num_bpoints = 20
    for k in range(5):
        print(np.max(fpop[:,:,k]*100))
        # get the maximum frontal frequency (rounded)
        maxPct = np.round(np.max(fpop[:,:,k][validHeights]*100)).astype(int)
        print(maxPct, flush = True)
        # chose points to have at least 600 samples (basepoints * bins), minimum 10 basepoints per bin 
        num_bpoints = max(10,(600//maxPct)+1)
        print(num_bpoints)
        maxPct+=1
        mySampleArray = np.zeros((maxPct, num_bpoints, 1000,5))
        myProbArray = np.zeros((maxPct,num_bpoints,1000))
        for p in range(0,maxPct,stepsize):
            print(p,end=": ")
            low_p = (p-stepsize) / 100
            up_p = p/100
            # get all values in the midlats that are within the percentile range
            # the first bin contains only the 0% frequencies
            if(p == 0):
                allPoss = np.nonzero(fpop[:,:,k]<=up_p)
            # all other bins contain the ]k-stepsize : k] % frequencies
            else:
                allPoss = np.nonzero((fpop[:,:,k]<=up_p) * (fpop[:,:,k]>(low_p)))
            # filter, to obtain only points that are also lower than the height threshold
            validPoss = np.nonzero(heightmap[allPoss])
            # build a tuple for indexing containing all points within the current percentile and simultaneously being lower than the height threshold
            allPos = (allPoss[0][validPoss], allPoss[1][validPoss])

            if(len(allPos[0])<num_bpoints):
                print("invalid p:", p, flush = True)
                mySampleArray[p,:,:,:] = np.NaN
                continue
            rand_points = rng.choice(len(allPos[0]), size=num_bpoints, replace = False)
            xsmpls = allPos[0][rand_points]
            ysmpls = allPos[1][rand_points]
            for point in range(num_bpoints):
                print(point, end=", ")
                xsmp = xsmpls[point]
                ysmp = ysmpls[point]
                myProbArray[p,point,:] = fpop[xsmp,ysmp,k]
                # get the hemisphere of the current bp
                north = xsmp < 161
                # define the range of the opposite hemisphere from the event mask
                fr = 420 if north else 100
                to = 581 if north else 261
                # The offset in the total fronts array
                # as fpop has the tropics removed its southern hemisphere has an additional 159 pixel offset (420-261)
                xoff = 100 if north else 259
                # get all points on the opposite hemisphere, where enough cases of extreme precipitation were identified
                hemPos = np.nonzero(evcount[fr:to]>= casePerPoint)
                for li in range(1000):
                    # draw random points from opposite hemisphere with enough precip events
                    rps = rng.choice(len(hemPos[0]), size = pointsPerList, replace=False)
                    # get the respective positions (xpos => add the offset to get the correct point in the 680x1400 mask)
                    xposs = hemPos[0][rps]+fr
                    yposs = hemPos[1][rps]
                    for rp in range(pointsPerList):
                        xpos = xposs[rp]
                        ypos = yposs[rp]
                        # get the count of events at this point
                        t_size = evcount[xpos, ypos]
                        # randomly select a start point, such that we can obtain case Per Point many samples
                        t_begin = rng.integers(low = 0, high=1+t_size-casePerPoint, size = 1)
                        # draw samples
                        t_list = np.nonzero(exEvs[:,xpos, ypos])[0][t_begin[0]:t_begin[0]+casePerPoint]
                        # for each timestamp read whether or not a front exists at the basepoint
                        mySampleArray[p, point, li] += np.sum(total_fronts[t_list, xsmp+xoff,ysmp].astype(np.int16), axis = 0) 
                    # turn in into a ratio
                    mySampleArray[p, point, li] /= casePerPoint*pointsPerList
            print(flush=True)

        
        with open(os.path.join("StatisticalTests",args.outname,"myRandSampResults_{}_{}.txt".format(tgt_season, k)), "w") as f:
            # calulate the numpy median and percentiles (just as info)
            for p in range(0,maxPct,stepsize):
                print(p, file = f)
                print(myProbArray[p,:,0], file = f)
                print(np.median(mySampleArray[p], axis=1), file = f)
                print(np.percentile(mySampleArray[p],1, axis=1), file = f)
                print(np.percentile(mySampleArray[p],99, axis=1), file = f)
            # reshape the Probability array (of all frontal probabilites of bp) into a 1d array
            ps = myProbArray.reshape(-1)
            # to get the intercept calculated
            ps = sm.add_constant(ps)
            # for each type of front calculate the 1 and 99 percentile using Quantile Regression
            for x in range(5):
                print("for x = ",x)
                # turn the mixed event in a 1d array as well
                exop = mySampleArray[::stepsize,:,:,x].reshape(-1)
                model1 = QuantReg(endog= exop, exog = ps, missing = 'drop')
                result1 = model1.fit(q=0.01, vcov = 'robust', kernel = 'epa', bandwidth = 'hsheather', max_iter = 1000, p_tol=1e-06)
                result2 = model1.fit(q=0.99, vcov = 'robust', kernel = 'epa', bandwidth = 'hsheather', max_iter = 1000, p_tol=1e-06)
                medi = model1.fit(q=0.5, vcov = 'robust', kernel = 'epa', bandwidth = 'hsheather', max_iter = 1000, p_tol=1e-06)
                print("0.01pct", result1.params, file = f)
                print("0.50pct", medi.params, file = f)
                print("0.99pct", result2.params, file = f)

        

if __name__ == "__main__":
    
    args = parseArguments()
    parOpt = None#setupDevice(args)

    name = os.path.join("StatisticalTests",args.outname)
    if(not os.path.isdir(name)):
        os.mkdir(name)
    tmpDataLoc = args.data
    if(not os.path.isdir(args.data)):
        args.data = os.path.dirname(args.data)
    data_set = setupDataset(args)    
    # 0 worker, to ignore problems caused by the multiprocessing
    loader = setupDataLoader(data_set, 0)
    # reset the correct data path
    args.data=tmpDataLoc
    
    num_samples = len(loader)
    if(args.num_samples != -1):
        num_samples = args.num_samples
    print("Evaluating {} Data files".format(num_samples))
    with torch.no_grad():
        performInference(loader, num_samples, parOpt, args)
        
