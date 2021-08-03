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
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm

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
    parser.add_argument("--tgtseason", type = str, default = "", help = "which season to evaluate for (djf, mam, jja, son), default no season")
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

    myEraExtractor = BinaryResultExtractor() 

    # Create Dataset
    data_set = WeatherFrontDataset(data_dir=data_fold, mapTypes = mapTypes, levelRange = myLevelRange, transform=myTransform, outSize=cropsize, labelThickness= labelThickness, era_extractor = myEraExtractor, has_subfolds = (False, False), asCoords = False, removePrefix = 0)
    return data_set




    

def setupDataLoader(data_set, args):
    # Create DataLoader 
    indices = list(range(len(data_set)))
    sampler = SequentialSampler(data_set)

    loader = DataLoader(data_set, shuffle=False, 
    batch_size = 1, sampler = sampler, pin_memory = True, 
    collate_fn = collate_wrapper(args.stacked, False, 0), num_workers = 8)
    return loader



def performInference(loader, num_samples, parOpt, args):
    border = 20
    
    # number of iterations of dilation
    Boxsize = 10

    data_set = loader.dataset
    no = data_set.removePrefix
    mapType = "hires" if args.NWS else "NA"

    tgtvar = ""
    if args.calcVar == "precip":
        front_file = "/lustre/project/m2_jgu-binaryhpc/Front_Detection_Data/PercentileData/tmp2016_frontprop.nc"
        tgtvar = "tp"
        mask_file = "/lustre/project/m2_jgu-binaryhpc/Front_Detection_Data/PercentileData/tmp2016_eventMask.nc"
        height_file = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/era5_const.nc"
    elif args.calcVar == "10mwind":
        pct_file = ""
        tgtvar = "u"
    elif args.calcVar == "10mwinddir":
        pct_file = ""
        tgtvar = "v"
    skip = 0
    total_fronts = np.zeros((num_samples, 680, 1400, 5), dtype=np.bool)
    all_fronts = np.zeros((360, 720, 5))
    if(False):
        for idx, data in enumerate(tqdm(loader, desc ='eval'), 0):
            if idx<skip:
                continue
            if(idx == num_samples+skip):
                exit(1)
                break
            inputs, labels, filename = data.data.cpu().numpy(), data.labels, data.filenames
            
            year,month,day,hour = filename[0][no:no+4],filename[0][no+4:no+6],filename[0][no+6:no+8],filename[0][no+9:no+11]
            #if(hour in ["06","12","18","00"]):
            #    all_fronts += inputs[0]
            inputs = inputs[:,border:-border,border:-border,:]

            monthid = int(month)-1
            front = inputs[0]
            for ftype in range(5):
                front[:,:,ftype] = distance_transform_edt(1-front[:,:,ftype])<=10
                # reate wide data
                #for _ in range(Boxsize):
                #    front[:,:,ftype] = morphology.binary_dilation(front[:,:,ftype])
                # sum the result to get the total front map
                #total_fronts[monthid, ftype]  += front[:,:,ftype]
            total_fronts[idx,:,:,:] = front[:,:,:].astype(np.bool)
        #all_fronts.tofile("compareAgainstClimatology.bin")
        total_fronts.tofile("/lustre/project/m2_jgu-binaryhpc/Front_Detection_Data/PercentileData/tmp2016_front4d_l2.bin")
    else:
        pass
        total_fronts = np.fromfile("/lustre/project/m2_jgu-binaryhpc/Front_Detection_Data/PercentileData/tmp2016_front4d_l2.bin", dtype=np.bool).reshape(num_samples,680,1400,5)
        #for x in range(num_samples):
        #    total_fronts[x,:,:,0] = np.sum(total_fronts[x,:,:,1:4],axis=-1).astype(np.bool)
        #total_fronts = np.flip(total_fronts, axis = 0)

    # Read the percentile mask
    #rootgrp = netCDF4.Dataset(os.path.realpath(front_file), "r", format="NETCDF4", parallel=False)
    #tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
    # the files have lat 90 - -90,   lon 0 - 360
    # => we need to offset lonrange
    #fpop = np.zeros((abs(int(tgt_latrange[0])-int(tgt_latrange[1]))*4, abs(int(tgt_lonrage[1])-int(tgt_lonrage[0]))*4))
    #if(tgt_lonrage[0] < 0 and tgt_lonrage[1] >= 0):
    #    fpop[:,:-int(tgt_lonrage[0])*4] =  rootgrp["tp"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:]
    #    fpop[:,-int(tgt_lonrage[0])*4:] = rootgrp["tp"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, :int(tgt_lonrage[1])*4]
    #else:
    #    fpop = rootgrp[tgtvar][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
    #rootgrp.close()
    
    # Read the event mask
    rootgrp = netCDF4.Dataset(os.path.realpath(mask_file), "r", format="NETCDF4", parallel=False)
    tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
    # the files have lat 90 - -90,   lon 0 - 360
    # => we need to offset lonrange
    exEvs = np.zeros((num_samples, abs(int(tgt_latrange[0])-int(tgt_latrange[1]))*4, abs(int(tgt_lonrage[1])-int(tgt_lonrage[0]))*4), dtype =np.bool)
    if(tgt_lonrage[0] < 0 and tgt_lonrage[1] >= 0):
        pass
        exEvs[:,:,:-int(tgt_lonrage[0])*4] =  (rootgrp["tp"][:num_samples,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:]).astype(np.bool)
        exEvs[:,:,-int(tgt_lonrage[0])*4:] = (rootgrp["tp"][:num_samples,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, :int(tgt_lonrage[1])*4]).astype(np.bool)
    else:
        pass
        exEvs = rootgrp[tgtvar][:,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
    rootgrp.close()



    # read the height map and remove all height +5 pixel radius (dilation)
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

    #get a filter map for height lower than 3000
    heightmap.tofile("before.bin")
    print(np.max(heightmap))
    # transform geopotential to height in m
    heightmap /= 9.80665
    heightmap.tofile("after.bin")
    print(np.max(heightmap))
    # get all points above 3000m
    heightmap = heightmap[border:-border,border:-border] >= 3000

    # dilate map by 5
    numHeightDilation = 5
    for _ in range(numHeightDilation):
        heightmap = morphology.binary_dilation(heightmap)

    #invert the map to get all points which are not considered a mountain
    heightmap = 1-heightmap
    heightmap.tofile("heightFilterMap.bin")

    #create the 
    #fpop = fpop[border:-border,border:-border] 
    exEvs = exEvs[:,border:-border,border:-border] 
    #get Equal ditribution or 1%bins
    rng = np.random.default_rng(12345)
    # 101 propabliliptes, 10 points, 1000 samples, 6 locations, 50 events)
    #mySamplePoints = np.zeros(101,10,2)
    #mySamplePos = np.zeros(101,10,1000, 6, 50)
    # generate the sample points
    '''
    for p in range(101):
        low_p = (p-1) / 10
        up_p = p/10
        if(p == 0):
            allPos = np.nonzero(fpop<=up_p)
        else:
            allPos = np.nonzero((fpop<=up_p) * (fpop>(low_p)))
        rand_points = rng.choice(len(allPos[0]), size=10, replace = False)
        mySamplePoints[p,:,0] = allPos[0][rand_points]
        mySamplePoints[p,:,1] = allPos[1][rand_points]
    '''


    dpm = [31,29,31,30,31,30,31,31,30,31,30,31]
    ssf = np.cumsum(np.array([0]+dpm))*24
    seasons = ["djf", "mam", "jja", "son"]
    tgt_season = args.tgtseason
    
    # when evaluating seasons, we should expect less extreme events per season
    casePerPoint = 13
    if(tgt_season == "djf"):
        evcount = np.sum(exEvs[ssf[11]:ssf[12]], axis=0)+np.sum(exEvs[:ssf[2]], axis= 0)
        fpop = (np.sum(total_fronts[ssf[11]:ssf[12]], axis=0)+ np.sum(total_fronts[:ssf[2]], axis=0))/((31+31+29)*24)
    elif(tgt_season == "mam"):
        evcount = np.sum(exEvs[ssf[2]:ssf[5]], axis = 0)
        fpop = np.sum(total_fronts[ssf[2]:ssf[5]], axis=0)/((31+30+31)*24)
    elif(tgt_season == "jja"):
        evcount = np.sum(exEvs[ssf[5]:ssf[8]], axis = 0)
        fpop = np.sum(total_fronts[ssf[5]:ssf[8]], axis=0)/((30+31+31)*24)
    elif(tgt_season == "son"):
        evcount = np.sum(exEvs[ssf[8]:ssf[11]], axis = 0)
        fpop = np.sum(total_fronts[ssf[8]:ssf[11]], axis=0)/((30+31+30)*24)
    else:
        evcount = np.sum(exEvs, axis=0)
        fpop = np.sum(total_fronts, axis=0)/num_samples
        casePerPoint = 50
    # generate the event lists
    # get the total count of extreme events
    # get the ratio of front events 
    
    k = 0
    stepsize = 1
    fpop = np.concatenate((fpop[100:260],fpop[420:580]),axis=0)
    heightmap = np.concatenate((heightmap[100:260], heightmap[420:580]), axis=0)
    validHeights = np.nonzero(heightmap)
    num_bpoints = 20
    for k in range(5):
        print(np.max(fpop[:,:,k]*100))
        maxPct = np.round(np.max(fpop[:,:,k][validHeights]*100)).astype(int)
        print(maxPct, flush = True)
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
            if(p == 0):
                allPoss = np.nonzero(fpop[:,:,k]<=up_p)
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
                north = xsmp < 160
                fr = 420 if north else 100
                to = 580 if north else 260
                xoff = 100 if north else 260
                hemPos = np.nonzero(evcount[fr:to]>= casePerPoint)
                for li in range(1000):
                    # draw random points from opposite hemisphere
                    rps = rng.choice(len(hemPos[0]), size = 6, replace=False)
                    xposs = hemPos[0][rps]+fr
                    yposs = hemPos[1][rps]
                    for rp in range(6):
                        xpos = xposs[rp]
                        ypos = yposs[rp]
                        t_size = evcount[xpos, ypos]
                        t_begin = rng.integers(low = 0, high=1+t_size-casePerPoint, size = 1)
                        t_list = np.nonzero(exEvs[:,xpos, ypos])[0][t_begin[0]:t_begin[0]+casePerPoint]
                        mySampleArray[p, point, li] += np.sum(total_fronts[t_list, xsmp+xoff,ysmp].astype(np.int16), axis = 0) 
                    mySampleArray[p, point, li] /= casePerPoint*6
                    #print(mySampleArray[p, point, li])
            print(flush=True)
#                        mySamplePos[p, point, li, rp, :] = t_list[:]

        

        #print(np.linalg.norm(np.sum(total_fronts, axis=0)-fpop))
        '''
        for p in range(101):
            xpos = mySamplePoints[p,:,0]
            ypos = mySamplePoints[p,:,1]
            for pt in range(10):
                for li in range(1000):
                    for rp in range(6):
                        mySampleArray[p,pt,li] += np.sum(total_fronts[xpos[pt], ypos[pt], mySamplePos[p,pt,li,rp]])
                    mySampleArray[p,pt,li] /= 50*6
        '''
        
        with open(args.outname+"/myRandSampResults_{}_{}.txt".format(tgt_season, k), "w") as f:
            for p in range(0,maxPct,stepsize):
                print(p, file = f)
                print(myProbArray[p,:,0], file = f)
                print(np.median(mySampleArray[p], axis=1), file = f)
                print(np.percentile(mySampleArray[p],1, axis=1), file = f)
                print(np.percentile(mySampleArray[p],99, axis=1), file = f)
            ps = myProbArray.reshape(-1)#np.array([[max(0,x-0.5)]*10*1000 for x in range(0,maxPct,stepsize)]).reshape(-1)
            ps = sm.add_constant(ps)
            for x in range(5):
                print("for x = ",x)
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
    parOpt = setupDevice(args)

    name = os.path.join("EffectAreas",args.outname)
    
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
    
    num_samples = len(loader)
    if(args.num_samples != -1):
        num_samples = args.num_samples
    print("Evaluating {} Data files".format(num_samples))
    with torch.no_grad():
        performInference(loader, num_samples, parOpt, args)
        
