from FrontPostProcessing import filterFronts
import numpy as np
import torch
import os

from skimage.io import imsave 
from skimage import measure, morphology


from torch.utils.data import DataLoader, SequentialSampler

#from MyLossFunctions import *

from Models.FDU3D import *

from tqdm import tqdm
import argparse

from era5dataset.FrontDataset import *
# ERA Extractors
from era5dataset.EraExtractors import *

from IOModules.csbReader import *

from NetInfoImport import *

from FrontPostProcessing import filterFronts, filterFrontsFreeBorder

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
    parser.add_argument('--outpath', default = ".", help='path to where the output shall be written')
    parser.add_argument('--outname', help='name of the output')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--device', type = int, default = 0, help = "number of device to use")
    parser.add_argument('--num_samples', type = int, default = -1, help='number of samples to infere from the dataset')
    parser.add_argument('--classes', type = int, default = 1, help = 'How many classes the network should predict (binary case has 1 class denoted by probabilities)')
    parser.add_argument('--normType', type = int, default = 0, help = 'How to normalize the data: 0 min-max, 1 mean-var, 2/3 the same but per pixel')
    parser.add_argument('--labelGroupingList', type = str, default = None, help = 'Comma separated list of label groups \n possible fields are w c o s (warm, cold, occluson, stationary)')
    parser.add_argument('--fromFile', type = str, default = None, help = 'file to extract network configuration from')
    parser.add_argument('--border', type = int, default = 5, help = "A Border in degree which is not evaluated")
    parser.add_argument('--skip', type = int, default = 0, help = "How many of the data should be skipped (skip + not skipped = num_samples)")
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
    cropsize = (720, 1440)
    mapTypes = {"NA": ("NA", (90,-89.75), (-180,180), (-0.25, 0.25), None) }

    myLevelRange = np.arange(105,138,4)

    myTransform = (None, None)
    labelThickness = 1
    labelTrans = (0,0)

    labelGroupingList = args.labelGroupingList

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
    # Create Dataset
    subfolds = (False, False)
    remPref = 0

    data_set = WeatherFrontDataset(data_dir=data_fold, label_dir=None, mapTypes = mapTypes, levelRange = myLevelRange, transform=myTransform, outSize=cropsize, labelThickness= labelThickness, label_extractor = None, era_extractor = myEraExtractor, asCoords = False, has_subfolds = subfolds, removePrefix = remPref)
    return data_set

def setupDataLoader(data_set, numWorkers):
    # Create DataLoader 
    sampler = SequentialSampler(data_set)
    loader = DataLoader(data_set, shuffle=False, 
    batch_size = 1, sampler = sampler, pin_memory = True, 
    collate_fn = collate_wrapper(True, False, 0), num_workers = numWorkers)
    return loader


def filterChannels(data, args):
    labelsToUse = args.labelGroupingList.split(",")
    possLabels = ["w","c","o","s"]
    for idx, possLab in enumerate(possLabels, 1):
        isIn = False
        for labelGroup in labelsToUse:
            if(possLab in labelGroup):
                isIn = True
        if(not isIn):
            data[0,:,:,0] -= data[0,:,:,idx]
    return data


def inferResults(model, inputs, args):
    outputs = model(inputs)
    outputs = outputs.permute(0,2,3,1)
    smoutputs = torch.softmax(outputs.data, dim = -1)
    smoutputs[0,:,:,0] = 1-smoutputs[0,:,:,0]

    # If some labels are not to be considered additionally remove them from the 0 case (all others don't matter)
    smoutputs = filterChannels(smoutputs, args)
    smoutputs = filterFronts(smoutputs.cpu().numpy(), args.border*4)
    return torch.from_numpy(smoutputs)

def performInference(model, loader, num_samples, parOpt, args):
    no = loader.dataset.removePrefix
    outfolder = os.path.join(args.outpath, "Detections")
    if(not os.path.isdir(args.outpath)):
        print("Could not find Output-Path, abort execution")
        print("Path was: {}".format(args.outpath))
        exit(1)
    if(not os.path.isdir(outfolder)):
        print("Creating Detection Folder at Output-Path")
        os.mkdir(outfolder)
    outname = os.path.join(outfolder, args.outname)
    if(not os.path.isdir(outname)):
        print("Creating Folder {} to store results".format(args.outname))
        os.mkdir(outname)
    for idx, data in enumerate(tqdm(loader, desc ='eval'), 0):
        if(idx == num_samples):
            break
        if(idx <= args.skip):
            continue
        inputs, labels, filename = data
        inputs = inputs.to(device = parOpt.device, non_blocking=False)
        # Create Results
        smoutputs = inferResults(model, inputs, args).numpy().astype(np.bool)
        smoutputs.tofile(os.path.join(outname,filename[0][no:]))

def setupModel(args, parOpt):
    model = None
    embeddingFactor = 6
    SubBlocks = (3,3,3)
    kernel_size = 5
    model = FDU2DNetLargeEmbedCombineModular(in_channel = args.in_channels, out_channel = args.out_channels, kernel_size = kernel_size, sub_blocks = SubBlocks, embedding_factor = embeddingFactor).to(parOpt.device)
    model.load_state_dict(torch.load(args.net, map_location = parOpt.device))
    model = model.eval()
    if(parOpt.myRank == 0):
        print()
        print("Begin Evaluation of Data...")
        print("Network:", model)
    return model


if __name__ == "__main__":
    
    args = parseArguments()
    parOpt = setupDevice(args)

    args.stacked = True
    data_set = setupDataset(args)    
    loader = setupDataLoader(data_set, 8)
    
    sample_data = data_set[0]
    data_dims = sample_data[0].shape
    print(data_dims)
    #label_dims = sample_data[1].shape


    # Data information
    in_channels = data_dims[0]
    levels = data_dims[0]
    latRes = data_dims[1]
    lonRes = data_dims[2]
    
    out_channels = args.classes
    if(args.binary):
        out_channels = 1
    args.in_channels = in_channels
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
    model = setupModel(args, parOpt)
    
    num_samples = len(loader)
    if(args.num_samples != -1):
        num_samples = args.num_samples

    print("Evaluating {} Data files".format(num_samples))
    with torch.no_grad():
        performInference(model, loader, num_samples, parOpt, args)
        
