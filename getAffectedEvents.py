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

from FrontPostProcessing import filterFronts
import netCDF4



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


def performInference(model, loader, num_samples, parOpt, args):
    border = 20
    
    # number of iterations of dilation
    Boxsize = 10

    Front_Event_count = 0
    Event_count = 0
    Extreme_Event_count = 0
    Front_Extreme_Event_count = 0

    data_set = loader.dataset
    no = data_set.removePrefix
    mapType = "hires" if args.NWS else "NA"

    tgtvar = ""
    if args.calcVar == "precip":
        pct_file = ""
        tgtvar = "tp"
    elif args.calcVar == "10mwind":
        pct_file = ""
        tgtvar = "u"
    elif args.calcVar == "10mwinddir":
        pct_file = ""
        tgtvar = "v"

    rootgrp = netCDF4.Dataset(os.path.realpath(pct_file), "r", format="NETCDF4", parallel=False)
    tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
    percentile_99 = rootgrp[tgtvar][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
    rootgrp.close()

    skip = 0
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
        
        year,month,day,hour = filename[0][no:no+4],filename[0][no+4:no+6],filename[0][no+6:no+8],filename[0][no+9:no+11]
        
        if args.calcVar == "precip":
            # Get Corresponding file with data
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/precip{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            var = rootgrp["tp"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            rootgrp.close()
        elif(args.calcVar == "10mwinddir"):
            grad = True
            orientation = True
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/B{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            u10dir = rootgrp["u10"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            v10dir = rootgrp["v10"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            var = np.angle(u10dir+1j*v10dir)
            rootgrp.close()
        elif(args.calcVar == "10mwind"):
            newFile = "/lustre/project/m2_jgu-w2w/ipaserver/ERA5/{0}/{1}/B{0}{1}{2}_{3}.nc".format(year,month,day,hour)
            rootgrp = netCDF4.Dataset(os.path.realpath(newFile), "r", format="NETCDF4", parallel=False)
            tgt_latrange, tgt_lonrage = data_set.getCropRange(data_set.mapTypes[mapType][1], data_set.mapTypes[mapType][2], data_set.mapTypes[mapType][3], 0)
            u10dir = rootgrp["u10"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            v10dir = rootgrp["v10"][0,int(90-tgt_latrange[0])*4:int(90-tgt_latrange[1])*4, int(tgt_lonrage[0])*4:int(tgt_lonrage[1])*4]
            var = np.abs(u10dir+1j*v10dir)
            rootgrp.close()

        percentile_99 = 1
        events = var > 0
        extreme_events = var > percentile_99
        front = outputs[0,:,:,0]
        # Count Events
        Event_count += len(np.nonzero(events)[0])
        # Count Extreme Events
        Extreme_Event_count += len(np.nonzero(extreme_events)[0])
        # Widen Fronts according to boxsize
        for _ in range(Boxsize):
            front = morphology.binary_dilation(front)
        # Count Events associated with a Front
        Front_Event_count += len(np.nonzero(events*front)[0])
        Front_Extreme_Event_count += len(np.nonzero(extreme_events*front)[0])
        print(Front_Event_count / Event_count)
        print(Front_Extreme_Event_count / Extreme_Event_count)

    return Event_count, Extreme_Event_count, Front_Event_count, Front_Extreme_Event_count


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
        EC, EEC, FEC, FEEC = performInference(model, loader, num_samples, parOpt, args)

    '''
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
    '''
        
