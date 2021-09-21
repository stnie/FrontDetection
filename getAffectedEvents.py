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

from skimage import morphology
from skimage.io import imsave
from FrontPostProcessing import filterFronts
import netCDF4
from InferOutputs import setupDataLoader, setupDevice, inferResults, DistributedOptions



def parseArguments():
    parser = argparse.ArgumentParser(description='FrontNet')

    parser.add_argument('--data', help='path to folder containing data')
    parser.add_argument('--precip', help='path to folder containing Precipitation Data')
    parser.add_argument('--pctFile', help='path to File containing the 99th percentile for thresholding of extreme events')
    parser.add_argument('--singleFile', type = int, help='data is a single file containing all fronts and precip folder containg a single file containing all extrem events named "tmp2016_eventMasks_<season>.nc"')
    parser.add_argument('--season', type = str, default = "all", help='season to calculate for (djf, mam, jja, son) , default whole year is take')
    parser.add_argument('--extremeInfluence', type = int, help='Use grid points associated with an extreme event instead of front, keeping fronts spatially accurate but dilating extreme events instead')
    parser.add_argument('--outname', help='name of the output')
    parser.add_argument('--num_samples', type = int, default = -1, help='number of samples to infere from the dataset')
    parser.add_argument('--preprocessed', type=int, help='fronts are already processed => no dilation necessary')
    parser.add_argument('--calcVar', type = str, default = "precip", help = 'which variable to measure along the cross section')
    args = parser.parse_args()
    
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



def readSecondary(rootgrp, var, time, latrange, lonrange):
    vals = np.zeros((abs(int((latrange[0]-latrange[1])*4)), abs(int((lonrange[1]-lonrange[0])*4))))
    if(lonrange[0] < 0 and lonrange[1] >= 0):
        vals[:,:-int(lonrange[0]*4)] =  rootgrp[var][time,int((90-latrange[0])*4):int((90-latrange[1])*4), int(lonrange[0]*4):]
        vals[:,-int(lonrange[0]*4):] = rootgrp[var][time,int((90-latrange[0])*4):int((90-latrange[1])*4), :int(lonrange[1]*4)]
    else:
        vals[:,:] = rootgrp[var][time,int((90-latrange[0])*4):int((90-latrange[1])*4), int(lonrange[0]*4):int(lonrange[1]*4)]
    return vals

def performInference(loader, num_samples, parOpt, args):
    border = 20
    
    # number of iterations of dilation
    Boxsize = 10

    Front_Event_count = np.zeros((5))
    Event_count = 0
    Extreme_Event_count = 0
    Front_Extreme_Event_count = np.zeros((5))

    data_set = loader.dataset
    no = data_set.removePrefix
    mapType = "all"

    tgtvar = ""
    if args.calcVar == "precip":
        pct_file =  args.pctFile
        tgtvar = "tp"
    else:
        print("variable not implemented. abort!")
        exit(1)

    # read the percentile 
    rootgrp = netCDF4.Dataset(os.path.realpath(pct_file), "r", format="NETCDF4", parallel=False)
    tgt_latrange, tgt_lonrange = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
    percentile_99 = readSecondary(rootgrp, tgtvar, 0, tgt_latrange, tgt_lonrange)
    rootgrp.close()



    skip = 0

    singleFiles = args.singleFile

    

    tgt_season = args.season
    tgt_mnths = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    if(tgt_season == "djf"):
        tgt_mnths = ["01","02","12"]
    elif(tgt_season == "mam"):
        tgt_mnths = ["03","04","05"]
    elif(tgt_season == "jja"):
        tgt_mnths = ["06","07","08"]
    elif(tgt_season == "son"):
        tgt_mnths = ["09","10","11"]

    num_months = len(tgt_mnths)
    # prepare output arrays
    
    total_events = np.zeros((num_months,percentile_99.shape[0]-2*border, percentile_99.shape[1]- 2*border))
    total_extreme_events = np.zeros_like(total_events)
    total_front_events = np.zeros((num_months,5,percentile_99.shape[0]-2*border, percentile_99.shape[1]- 2*border))
    total_front_extreme_events = np.zeros_like(total_front_events)
    total_fronts = np.zeros_like(total_front_events)
    print("singleFiles is {} from {}".format(singleFiles, args.singleFile))
    # Fronts and Precipitation are in single files per timestep. The script will gradually build the extrem event mask and aggregate the results
    if(singleFiles):
        for idx, data in enumerate(tqdm(loader, desc ='eval'), 0):
            if idx<skip:
                continue
            if(idx == num_samples+skip):
                break
            inputs, labels, filename = data.data.cpu().numpy(), data.labels, data.filenames

            # skip infer if wrong month is drawn
            year,month,day,hour = filename[0][no:no+4],filename[0][no+4:no+6],filename[0][no+6:no+8],filename[0][no+9:no+11]
            if(not (month in tgt_mnths)):
                continue

            outputs = inputs
            
            # Get Corresponding file with precipitation data
            precipFile = os.path.join(args.precip, year, month, "precip{0}{1}{2}_{3}.nc".format(year,month,day,hour))
            rootgrp = netCDF4.Dataset(os.path.realpath(precipFile), "r", format="NETCDF4", parallel=False)
            var = readSecondary(rootgrp, tgtvar, 0, tgt_latrange, tgt_lonrange)
            rootgrp.close()

            monthID = tgt_mnths.index(month)
            # All events mask (value > 0)
            events = (var > 0)
            # All extreme events mask 
            extreme_events = var > percentile_99
            if(args.extremeInfluence):
                extreme_events = distance_transform_edt(1-extreme_events) <= Boxsize
                events = distance_transform_edt(1-events) <= Boxsize
            # crop the outer area which is not correctly predicted
            events = events[border:-border, border:-border]
            extreme_events = extreme_events[border:-border, border:-border]
            # Aggregated events
            total_events[monthID] += events*1
            # Aggregated extreme Events
            total_extreme_events[monthID] += extreme_events*1
            # Count Events
            Event_count += np.sum(events)
            # Count Extreme Events
            Extreme_Event_count += np.sum(extreme_events)
            # for each type of front
            for ftype in range(outputs.shape[-1]):
                front = outputs[0,border:-border,border:-border,ftype]
                if(not args.extremeInfluence):
                    # Widen Fronts according to boxsize
                    for ftype in range(5):
                        front[:,:,ftype] = distance_transform_edt(1-front[:,:,ftype])<=Boxsize

                # Count Events associated with a Front
                front_events = events*front
                front_extreme_events = extreme_events * front

                Front_Event_count[ftype] += np.sum(front_events)
                Front_Extreme_Event_count[ftype] += np.sum(front_extreme_events)
                

                total_front_events[monthID, ftype] += front_events*1
                total_front_extreme_events[monthID, ftype] += front_extreme_events*1
                total_fronts[monthID, ftype] += front*1
    else:
        # all Fronts are in a single file, all extrem events are a single file. The script will simply aggregate the results
        assert(not os.path.isdir(args.data))
        extreme_Influence = args.extremeInfluence
        # read the undilated file instead
        if(not args.preprocessed):
            frontFile = args.data
            allFrontEvents = np.fromfile(frontFile, dtype=np.bool).reshape(-1, 720, 1440, 5)
            allFrontEvents = allFrontEvents[:,border:-border,border:-border]
            if(not extreme_Influence):
                for x in range(allFrontEvents.shape[0]):
                    print("Widening front Event {}".format(x), flush = True)
                    for ft in range(allFrontEvents.shape[-1]):
                        allFrontEvents[x,:,:,ft] = distance_transform_edt(1-allFrontEvents[x,:,:,ft]) <= Boxsize
            #allFrontEvents.astype(np.bool).tofile(os.path.join(args.precip, "widenedFronts2016_{}_l2norm_{}.bin".format(args.season, Boxsize)))
            #exit(1)
        else:
            frontFile = args.data
            allFrontEvents = np.fromfile(frontFile, dtype=np.bool).reshape(-1, 680, 1400, 5)

            
        num_samples = allFrontEvents.shape[0]
        # just for lookings
        for x in range(5):
            imsave("mytest{}.png".format(x), allFrontEvents[0,:,:,x])
        # load the map with all precipitation extrem events
        
        precipFile = os.path.join(args.precip, "tmp2016_eventMask_{}.nc".format(args.season))
        rootgrp = netCDF4.Dataset(os.path.realpath(precipFile), "r", format="NETCDF4", parallel=False)
        num_samples = rootgrp["time"][:].shape[0]
        tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
        tgtType = np.bool
        allExtremeEvents = np.zeros((num_samples, 720, 1440), dtype=tgtType)
        print("num_samples for season {} is {}".format(args.season, num_samples))
        if(tgt_lonrage[0] < 0 and tgt_lonrage[1] >= 0):
            allExtremeEvents[:,:,:-int(tgt_lonrage[0])*4] =  rootgrp["tp"][:num_samples,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:].astype(tgtType)
            allExtremeEvents[:,:,-int(tgt_lonrage[0])*4:] = rootgrp["tp"][:num_samples,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, :int(tgt_lonrage[1])*4].astype(tgtType)
        else:
            allExtremeEvents = rootgrp["tp"][:num_samples,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4].astype(tgtType)
        # we need to find all points that are influenced by the extreme event instead => l2 norm dilation is necessary
        if(extreme_Influence):
            for x in range(num_samples):
                print("Widening extreme Event {}".format(x), flush = True)
                allExtremeEvents[x] = distance_transform_edt(1-allExtremeEvents[x]) <= Boxsize
        imsave("mytestextremes.png", allExtremeEvents[0])
        allExtremeEvents= allExtremeEvents[:,border:-border,border:-border]

        # create the count maps
        dpm = np.array([0,31,29,31,30,31,30,31,31,30,31,30,31])
        dps = dpm*1
        ssf = np.cumsum(dpm)*24
        tgt_season = args.season 
        if(tgt_season == "djf"):
            #the file orders it chronologically so its jf d (also for the event mask)
            allFrontEvents = np.concatenate((allFrontEvents[:ssf[2]], allFrontEvents[ssf[11]:ssf[12]]), axis=0)
            dps =np.array([0,31,29,31])
        elif(tgt_season == "mam"):
            allFrontEvents = allFrontEvents[ssf[2]:ssf[5]]
            dps =np.array([0,31,30,31])
        elif(tgt_season == "jja"):
            allFrontEvents = allFrontEvents[ssf[5]:ssf[8]]
            dps =np.array([0,30,31,31])
        elif(tgt_season == "son"):
            allFrontEvents = allFrontEvents[ssf[8]:ssf[11]]
            dps =np.array([0,30,31,30])
        tsf = np.cumsum(dps)*24
        print("all data loaded")
        # Go through all months in the season
        for m in range(len(dps)-1):
            print("front and extreme for month {}".format(m))
            for ft in range(5):
                total_fronts[m,ft] = np.sum(allFrontEvents[tsf[m]:tsf[m+1],:,:,ft], axis = 0)
                total_extreme_events[m] = np.sum(allExtremeEvents[tsf[m]:tsf[m+1]], axis=0)
        # Create the extreme and Front case
        for ft in range(5):
            allFrontEvents[:,:,:,ft] *= allExtremeEvents
        # And store it into the global array
        for m in range(len(dps)-1):
            print("front with extreme for month {}".format(m))
            for ft in range(5):
                total_front_extreme_events[m,ft] = np.sum(allFrontEvents[tsf[m]:tsf[m+1],:,:,ft], axis = 0)
        Extreme_Event_count = np.sum(total_extreme_events, axis= 0)
        Front_Extreme_Event_count = np.sum(total_front_extreme_events, axis= 0)


    return [Event_count, Extreme_Event_count, Front_Event_count, Front_Extreme_Event_count], [total_events, total_extreme_events, total_front_events, total_front_extreme_events, total_fronts]


if __name__ == "__main__":
    
    args = parseArguments()
    parOpt = None

    name = os.path.join("EffectAreas",args.outname)
    # In case of a single file, we need a directory to enable data_set creation
    tmpDataLoc = args.data
    if(not os.path.isdir(args.data)):
        args.data = os.path.dirname(args.data)
    data_set = setupDataset(args)    
    loader = setupDataLoader(data_set, 0)
    # reset the correct data path
    args.data=tmpDataLoc
    
    num_samples = len(loader)
    if(args.num_samples != -1):
        num_samples = args.num_samples
    print("Evaluating {} Data files".format(num_samples))
    with torch.no_grad():
        counts, images = performInference(loader, num_samples, parOpt, args)

    # Save everything
    EC, EEC, FEC, FEEC = counts
    Ei, EEi, FEi, FEEi, Fi = images
    if(not os.path.isdir(name)):
        os.mkdir(name)
    En = os.path.join(name, "events_"+args.season)
    EEn = os.path.join(name, "extreme_events_"+args.season)
    FEn = os.path.join(name, "front_events_"+args.season)
    FEEn = os.path.join(name, "front_extreme_events_"+args.season)
    Frn = os.path.join(name, "fronts_"+args.season)

    Ei.tofile(En+".bin")
    EEi.tofile(EEn+".bin")
    FEi.tofile(FEn+".bin")
    FEEi.tofile(FEEn+".bin")
    Fi.tofile(Frn+".bin")

        
