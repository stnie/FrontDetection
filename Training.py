import numpy as np
import argparse
import os

# torch
import torch

# Data loading
from torch.utils.data import DataLoader, SubsetRandomSampler

# Dat set
from era5dataset.FrontDataset import *
# ERA Extractors
from era5dataset.EraExtractors import *

# networks
from Models.FDU3D import *
from MyLossFunctions import *

# Augmentation
from torchvision.transforms import Compose
from MyTransformations import *

# distributed torch
from torch.nn.parallel import DistributedDataParallel as DDP

from datetime import timedelta

from ParallelStarter import BeginMultiprocessing

from IOModules.csbReader import *



class DistributedOptions():
    def __init__(self):
        self.myRank = -1
        self.device = -1
        self.local_rank = -1
        self.world_size = -1
        self.nproc_per_node = -1
        self.nnodes = -1
        self.node_rank = -1


def estimateWeights(dataLoader, sampleSize, labels, myRank, asCoords, args):
    return torch.ones(labels)

def train(model, train_loader, epoch, seasons, criterion, optimizer, args, parOpt, outputModifier = None):
    model.train()
    for season in range(seasons):
        running_loss = torch.cuda.FloatTensor(1).fill_(0)
        for i, data in enumerate((train_loader), 0):
            if(args.verbose):
                print("\rRank {}: Batch {}/{}".format(parOpt.myRank, i, len(train_loader)), end='')
            inputs, labels, names, masks = data

            inputs = inputs.to(device = parOpt.device, non_blocking=True)
            if(not args.elastic):
                labels = labels.to(device = parOpt.device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            if(args.deeplab):
                outputs = outputs['out']
            outputs = outputs.permute(0, 2, 3, 1)
            m = outputs.shape[0]
            if(outputModifier is not None):
                outputs = outputModifier(outputs)
            loss = criterion(outputs, labels, masks)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss += loss
        # output after each season
        with torch.no_grad():
            print('\rRank {} [Ep {}, Se {}] loss: '.format(parOpt.myRank,
                    epoch + 1, season + 1, ),100*running_loss / len(train_loader), "%")
    return running_loss/(len(train_loader))

def validate(model, test_loader, epoch, criterion, name, args, parOpt, outputModifier = None, save_intervall = 1):
    model.eval()
    distanceFunc = args.distance
    test_loss = torch.cuda.FloatTensor(1).fill_(0)
    total = 0.0001
    for idx, data in enumerate(test_loader, 0):
        if(args.verbose):
            print("\rRank {}: Batch {}/{}".format(parOpt.myRank, idx, len(test_loader)), end='')
        inputs, labels, names, masks = data
        
        inputs = inputs.to(device = parOpt.device, non_blocking=False)
        if(not args.elastic):
            labels = labels.to(device = parOpt.device, non_blocking=False)
        outputs = model(inputs)
        if(args.deeplab):
            outputs = outputs['out']
        
        outputs = outputs.permute(0, 2, 3, 1)
        m = outputs.shape[0]
        latRes = outputs.shape[1]
        lonRes = outputs.shape[2]

        out_channels = outputs.shape[-1]

        test_loss += criterion(outputs, labels, masks)

        total += 1
    print('Rank {}: Loss of the network on the test images: {} %'.format(parOpt.myRank, 100*test_loss/total))
    return test_loss/total

def save_checkpoint(model, optimizer, path, epoch, loss, weight):
    print("Saving checkpoint to {}".format(path[:-4]+str(epoch)+path[-4:]))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'weights': weight
        }, path[:-4]+str(epoch)+path[-4:])
            

def save_model(model, path):
    torch.save(model.state_dict(), path)        


def createDataLoader(args, transpose_rate, data_set, test_data_set, swap_indices, batchsize, parOpt):
    stacked = args.stacked
    elastic = args.elastic
    #### CREATE DATA LOADER ####
    my_collate_wrapper = collate_wrapper(stacked,  asCoordinates = elastic, transpose_rate = transpose_rate, swap_indices = swap_indices)

    # The data_set on this Node
    data_set_size = len(data_set)
    # ADJUST FOR MULTIPLE GPUS
    data_set_batch = data_set_size//parOpt.nproc_per_node
    possible_indices = np.random.permutation(data_set_size)
    #each local worker should get random samples from the samples on the node, prevent edge worker to be limited in their data sets
    train_indices = possible_indices[parOpt.local_rank*data_set_batch: (parOpt.local_rank+1)*data_set_batch]
    #train_indices = list(range(parOpt.local_rank*data_set_batch,(parOpt.local_rank+1)*data_set_batch))
    test_indices = list(range(len(test_data_set)))
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(data_set, shuffle=False, 
    batch_size = batchsize, sampler = train_sampler, pin_memory = True, collate_fn = my_collate_wrapper, num_workers = 8)

    test_loader = DataLoader(test_data_set, shuffle=False, 
    batch_size = 1, sampler = test_sampler, pin_memory = True,  collate_fn = my_collate_wrapper, num_workers = 8)
    return train_loader, test_loader


def setupDataset(args, cropSize, labelCropSize, globalLock, parOpt):
    data_location = args.root
    labelGrouping = args.labelGroupingList
    normType = args.normType
    distanceFunc = args.distance
    types = args.trainType

    # My Map Types
    myTypes = ["NA", "NT", "hires"]
    foldNames = ["NA", "NT", "hires"]
    # NA to NA, NT to NT, NWS to NWS. A more conservative range of lons and lats to reduce points not present in the labeled data 
    degBorder = args.border/4
    myLatsSafe = [(70+degBorder, 35-degBorder), (90-degBorder, 30+degBorder), (70+degBorder, 35-degBorder)]
    myLonsSafe = [(-45-degBorder, 35+degBorder), (-80-degBorder, 40+degBorder), (-135-degBorder, -60+degBorder)]
    #masks = [np.fromfile("Masks/NAMask.bin", dtype=np.float32).reshape(720,1440), None, np.fromfile("Masks/hiresMask.bin", dtype=np.float32).reshape(720,1440)]
    resolution = [(-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25)]

    myLevelRange = np.arange(105,138,4)

    # Create a default type, using the NA projection
    # key -> "fileIdentifier, latrange, lonrange, resolution"
    if(types == 0):
        # Train with Both
        myMapTypes = {myTypes[i]: (foldNames[i],myLatsSafe[i], myLonsSafe[i], resolution[i], None) for i in range(0,3,2)}
    elif(types == 1):
        # Train only NA
        myMapTypes = {myTypes[i]: (foldNames[i],myLatsSafe[i], myLonsSafe[i], resolution[i], None) for i in range(0,1,2)}
    elif(types == 2):
        # Train only hires
        myMapTypes = {myTypes[i]: (foldNames[i],myLatsSafe[i], myLonsSafe[i], resolution[i], None) for i in range(2,3,2)}
    print(myMapTypes) 
    # INPUT/OUTPUT SIZES
    outsizes = cropSize

    # Label can be translated by up to x,y pixel in lat,lon direction
    labelTrans = (0,0)
    hflip = RandomHorizontalFlip(0.5)
    vflip = RandomVerticalFlip(0.5)
    hcflip = RandomHorizontalCoordsFlip(size = labelCropSize, p = 0.5)
    vcflip = RandomVerticalCoordsFlip(size = labelCropSize, p = 0.5)
    myTransform = Compose([hflip, vflip])
    mylTransform = Compose([hcflip, vcflip])
    mytestTransform = Compose([hflip, vflip])
    mytestlTransform = Compose([hcflip, vcflip])
        
    transforms = (myTransform, mylTransform)
    testTransforms = (mytestTransform, mytestlTransform)

    # train data set
    data_fold = os.path.join(data_location,'data')
    label_fold = os.path.join(data_location,'label')

    # test data set
    test_data_fold = os.path.join(data_location,'test_data')
    test_label_fold = os.path.join(data_location,'test_label')

    myLineGenerator = extractCoordsInRange(labelGrouping)
    myLabelExtractor = DefaultFrontLabelExtractor(myLineGenerator)

    variables = ['t','q', 'u', 'v', 'w', 'sp', 'kmPerLon']
    
    # simple normalization (use estimates for max and min to normalize into -1 to 1 range)
    #myEraExtractor = DefaultEraExtractor(variables)
    # variance normalization (use estimates for mean and var to normalize to mean = 0 and var = 1)
    # determine which variables need their sign flipped if the input is horizontally or vertically flipped
    # generally: all kinds of derivatives need a flip
    horizontal_flips = [3]
    vertical_flips = [2]
    levels = myLevelRange.shape[0]
    
    myEraExtractor = DerivativeFlippingAwareEraExtractor(variables, horizontal_flips, vertical_flips, 0.5, 0 , 1, normType = normType, sharedObj = None)

    data_set = WeatherFrontDataset(data_dir=data_fold, label_dir=label_fold, mapTypes=myMapTypes, levelRange = myLevelRange, transform=transforms, outSize= outsizes, labelThickness = 1, label_extractor = myLabelExtractor, asCoords=args.elastic, era_extractor = myEraExtractor)
    test_data_set = WeatherFrontDataset(data_dir=test_data_fold, label_dir=test_label_fold, mapTypes=myMapTypes, levelRange = myLevelRange, transform=testTransforms, outSize= outsizes, labelThickness = 1, label_extractor = myLabelExtractor, asCoords=args.elastic, era_extractor = myEraExtractor)

    if(parOpt.myRank == 0):
        print()
        print("Training Data...")
        print("Data-Location:", data_fold)
        print("Label-Location:", label_fold)
        print()
        print("Test Data...")
        print("Data-Location:", test_data_fold)
        print("Label-Location:", test_label_fold)
        print()
        print("Augmentation:", myTransform)
        print("Using Variables:", variables)
        print()
        print("Era Extractor:", myEraExtractor)
        print("Using NormType:", normType)
        print("Front Extractor:", myLabelExtractor, myLabelExtractor.imageCreator)
        print("Used Map Types:", myMapTypes)
        print("Used Levels:", myLevelRange)
    return data_set, test_data_set, None


def createModel(in_channels, out_channels, args, parOpt, IMD_PATH):
    embeddingFactor = 6
    SubBlocks = (3,3,3)
    kernel_size = 5
    model = FDU2DNetLargeEmbedCombineModular(in_channel = in_channels, out_channel = out_channels, kernel_size = kernel_size, sub_blocks = SubBlocks, embedding_factor = embeddingFactor)

    # load the model if told so
    if(args.loadCP is not None):
        checkpoint = torch.load(IMD_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("loaded model state dict")

    # move the model to the gpu
    model = model.float().to(device = parOpt.device)
    
    if(args.distributed):
        ddp_model = DDP(model, device_ids=[parOpt.device], output_device=parOpt.device)
        return ddp_model, model
    
    return model, model


def getLabelWeight(args, IMD_PATH, test_loader, num_samples, out_channels, parOpt):
    if(args.loadCP is not None):
        print("Loading label weights...")
        checkpoint = torch.load(IMD_PATH)
        w = checkpoint["weights"]
        w.to(device = parOpt.device)
    else:
        print("Estimating label weights...")
        w = estimateWeights(test_loader, num_samples, out_channels, parOpt.myRank, args.elastic, args).to(device = parOpt.device)
        w.to(device = parOpt.device)
    if(args.distributed):
        torch.distributed.all_reduce(w)
        w /= parOpt.world_size
    return w

def getCriterion(args,w, parOpt):
    print("detection vs classification", args.weight,"vs", 1-args.weight)
    criterion = IoUWithFitRaw(weight = args.weight, interChannelWeight = w[1:], maxDist = 3, sigma = 3, deletion_error = -20000, channel_change = False, border = args.border)
    test_criterion = IoUWithFitRaw(weight = args.weight, interChannelWeight = w[1:], maxDist = 3, sigma = 3, deletion_error = -20000, channel_change = False, border = args.border)
    return criterion, test_criterion

def setupParallelTraining(local_rank, args):
    # PARALLEL LEARNING SECTION
    # These are the parameters used to initialize the process group
    # Print them out to check if parallelization works
    # additionally set world_size accordingly for learning rate and batchsize adjustments
    
    world_size = args.nnodes * args.nproc_per_node
    # global rank
    myRank = args.nproc_per_node* args.node_rank+local_rank
    
    # Set parallel Options
    parOpt = DistributedOptions()
    parOpt.nnodes = args.nnodes
    parOpt.nproc_per_node = args.nproc_per_node
    parOpt.node_rank = args.node_rank
    parOpt.world_size = world_size
    parOpt.myRank = myRank
    parOpt.local_rank = local_rank

    if(args.distributed):
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE")
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    # Set devices for each rank and initialize distributed training
    if not args.disable_cuda and torch.cuda.is_available():
        if(args.distributed):
            if(args.slurm):
                print("Begin Waiting for init node_rank {} rank {} on node {}".format(args.node_rank, local_rank, os.environ["SLURMD_NODENAME"]), flush=True)
                parOpt.device = args.available_gpu[local_rank]
            else:
                print("Begin Waiting for init node_rank {} rank {} on node {}".format(args.node_rank, local_rank, "unknown"), flush=True)
                parOpt.device = local_rank
            torch.cuda.set_device(parOpt.device)
            torch.distributed.init_process_group(backend='nccl', world_size = world_size, rank = args.nproc_per_node* args.node_rank + local_rank, init_method="env://", timeout=timedelta(minutes=30))
            assert(parOpt.myRank == torch.distributed.get_rank())
            print("initialized Rank (global, local):", parOpt.myRank, parOpt.local_rank)
        else:
            torch.cuda.set_device(int(args.device))
            parOpt.device = torch.device('cuda')
    else:
        print("Rank {} (global {}) has disabled Cuda {}:".format(local_rank, myRank, args.disable_cuda))
        print("Rank {} (global {}) has cuda available {}:".format(local_rank, myRank, torch.cuda.is_available()))

        parOpt.device = torch.device('cpu')
    return parOpt



def setupOutputModifier(args, parOpt):
    train_output_modifier = None
    test_output_modifier = None
    return train_output_modifier, test_output_modifier


def runTraining(local_rank, args, globalLock):
    parOpt = setupParallelTraining(local_rank, args)
    world_size = parOpt.world_size
    myRank = parOpt.myRank

    name = os.path.join(args.root,args.outname)
    PATH = name+str(args.trainType)+'.pth'
    if(args.loadCP is not None):
        IMD_PATH = name+str(args.trainType)+'_checkpoint'+str(args.loadCP)+'.pth'
    else:
        IMD_PATH = name+str(args.trainType)+'_checkpoint.pth'

    out_channels = args.classes

    #make sure that classes fits to the provided groups
    channelVsLabel = out_channels - len(args.labelGroupingList.split(","))
    # if the value is 0, the network predicts only the provided labels
    # if the value is 1, the network additionally provides a background channel, which corresponds to the absence of labels
    assert(channelVsLabel == 1 or channelVsLabel == 0)
    # Only a true / false label
    args.stacked = True

    # SETTABLE PARAMETER FOR TRAINING 
    # Data Parameter
    # Setup the crop => Input and Output of the network
    cropSize = (128,256)
    labelCropSize = cropSize

    # A transpose rate for each individual image loaded
    transpose_rate = 0.0
    
    # Training parameter
    batchsize = 16
    epochs = 10000
    seasons = 1
    # Number of epochs without improvement that are tolerated
    noImprovementMax = 20

    # initial Learning Rate
    initial_lr = 0.005
    # distributed training cannot change lr during training, so start with a lower lr by default
    if(args.distributed):
        initial_lr = 0.005

    save_intervall = 100   #checkpoints
    complete_save_intervall = 10    #whole model
    test_intervall = 10    # test intervall
    img_save_intervall = 5 # 1 = write images after each test phase

    # SETUP THE DATASET
    data_set, test_data_set, swap_indices = setupDataset(args, cropSize, labelCropSize,globalLock, parOpt)

    #### DATA_LOADER EXTRACTED TRAINING PARAMETER####
    test_sample = data_set[0]
    data_dims = test_sample[0].shape
    #label_dims = test_sample[1].shape


    # Number of channels to feed the network    
    in_channels = data_dims[0]

    # Resolution information on the output of the network 
    latRes = data_dims[1]
    lonRes = data_dims[2]

    
    if(myRank == 0):
        print("Datalayout...")
        print("Resolution in (after crop):", data_dims[-2], data_dims[-1])
        print("Resolution out (after crop):", latRes, lonRes)
        print("Channels:", in_channels)
        print("Labeltypes:", out_channels)
        print("")

    #### TRAINING PART #####

    ##### CREATE MODEL ######
    train_output_modifier, test_output_modifier = setupOutputModifier(args, parOpt)
    
    model, base_model = createModel(in_channels, out_channels, args, parOpt, IMD_PATH)

    ##### CREATE DATA LOADER #####
    train_loader, test_loader = createDataLoader(args, transpose_rate, data_set, test_data_set, swap_indices, batchsize, parOpt)
    #### ESTIMATE LABEL WEIGHT ####

    weight = getLabelWeight(args, IMD_PATH, test_loader, len(test_loader), out_channels, parOpt)
    
    #### SELECT LOSS FUNCTION ####
    criterion, test_criterion = getCriterion(args, weight, parOpt)
    #### SETUP OPTIMIZER ####
    # LR is scaled by world_size as gradient is averaged between nodes 
    optimizer = torch.optim.SGD(model.parameters(), lr = initial_lr * world_size, momentum = 0.9, weight_decay = 0, nesterov=True)
    
    # Write Dataset information to file
    if(myRank == 0):
        if(not os.path.isdir(name)):
            os.mkdir(name)
        text_file = open(os.path.join(name,"data_set_info.txt"), "w")
        n = text_file.write(data_set.getInfo())
        text_file.close()
        # Write trainings information
        text_file = open(os.path.join(name, "training_info.txt"), "w")
        n = text_file.write(str([weight, criterion, test_criterion, optimizer, initial_lr, batchsize, epochs, noImprovementMax ]))
        text_file.close()
    
    #### SET STARTING EPOCH ####
    sepoch = 0

    # load optimizer, starting epoch and loss
    if(args.loadCP is not None):
        checkpoint = torch.load(IMD_PATH)
        if(myRank == 0):
            pass
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        sepoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("loaded state dict")
        print("continuing at epoch {} with loss {:6.2f}%".format(sepoch, loss.data[0]))

    #### SET LR SCHEDULER ####
    # setup the optimizer scheduler to adjust learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # Distributed Training does not support change of LR => scheduler is not allowed
    if(args.distributed):
        scheduler = None
    
    #### PERFORM TRAINING AND EVALUATION ####

    if(myRank == 0):
        print()
        print("Begin Training of Data...")
        print("Network:", model)
        print("{} Data files".format(len(train_loader)+ len(test_loader)))
        print("{} Training files , {} Test files".format(len(train_loader),len(test_loader)))
        print("Batchsize:", batchsize)
        print("Epochs:", epochs)
        print("Seasons:", seasons)
        print("Loss Function:", criterion)
        print("Weights", weight)
        print("Optimizer:", optimizer)
        print("Training checkpoint intervall:", save_intervall)
        print("Model save intervall:", complete_save_intervall)
        text_file = open(os.path.join(name,"network_info.txt"), "w")
        n = text_file.write(str(model))
        text_file.close()
        

    # Can be used for plotting (not yet working)
    myTrainError = []
    myLR = []
    myTestError = []
    # initialize current_best_error with a high positive value
    current_best_loss = 100000
    NoImprovementSince = 0
    for epoch in range(sepoch, epochs):
        # Training Step
        running_loss = train(model, train_loader, epoch, seasons, criterion, optimizer, args, parOpt, train_output_modifier)
        myTrainError.append(running_loss.cpu().numpy())
        # LR Schedule Step
        if(scheduler is not None):
            oldLR = [group['lr'] for group in optimizer.param_groups]
            scheduler.step(running_loss)
            newLR = [group['lr'] for group in optimizer.param_groups]
            print("LR changed from {} to {}".format(oldLR, newLR))

        # Testing, Checkpointing and Saving
        with torch.no_grad():
            # Only Rank 0 should save the model
            if parOpt.myRank == 0:
                if(epoch % complete_save_intervall == complete_save_intervall-1):
                    save_model(base_model, PATH)
                if(epoch % save_intervall == save_intervall-1):
                    save_checkpoint(model, optimizer, IMD_PATH, epoch, running_loss, weight)
            # Test the current model
            if(epoch%test_intervall==test_intervall-1):
                train_loss = validate(model, test_loader, epoch, test_criterion, name, args, parOpt, test_output_modifier, img_save_intervall)
                if(args.distributed):
                    torch.distributed.all_reduce(train_loss)
                    train_loss /= parOpt.world_size
                    print("Average Train Loss: ", train_loss)
                if(train_loss < current_best_loss):
                    if(parOpt.myRank == 0):
                        save_model(base_model, os.path.splitext(PATH)[0]+"currBest.pth")
                    current_best_loss = train_loss
                    NoImprovementSince = 0
                else:
                    NoImprovementSince += 1
                myTestError.append(train_loss.cpu().numpy())
                if(NoImprovementSince >= noImprovementMax//2):
                    print("divide LR by 10")
                    for g in optimizer.param_groups:
                        g['lr'] = max(g['lr']/10, 0.0000001)
                    for g in optimizer.param_groups:
                        if(g['lr'] > 0.0000001):
                            NoImprovementSince = 0
                if(NoImprovementSince >= noImprovementMax):
                    print("no improvement since", noImprovementMax,"epochs")
                    break
    # Cleanup
    # Save the final model and end Multiprocessing
    if myRank == 0:
        print("Stopped after", epoch, "epochs")
        if(args.distributed):
            save_model(base_model, os.path.splitext(PATH)[0]+"tmp.pth")
        else:
            save_model(model, PATH)
        np.array(myTrainError).tofile(name+"/type"+str(args.trainType)+"trainError.bin")
        np.array(myTestError).tofile(name+"/type"+str(args.trainType)+"testError.bin")
        
    if args.distributed:
        torch.distributed.destroy_process_group()

def ParseArguments(parser):
    parser.add_argument('--outname', type = str, help='name of the output')
    parser.add_argument('--root', type = str, help = 'root of the data')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--loadCP', type=str, default=None, help='Use a pretrained model to resume training')
    parser.add_argument('--elastic', action='store_true', help='Use elastic fit before loss')
    parser.add_argument('--border', type = int, default = 0, help ="Padding for the extraction, that is ignored in the loss (increases input dimension!)")
    parser.add_argument('--device', type = int, default=0, help = "number of device to use")
    parser.add_argument('--deeplab', action='store_true', help='Use Deeplab architecture from torch prebuilt networks')
    parser.add_argument('--classes', type = int, default = 1, help = 'How many classes the network should predict (binary case has 1 class denoted by probabilities)')
    parser.add_argument('--labelGroupingList', type = str, default = None, help = 'Comma separated list of label groups \n possible fields are w c o s (warm, cold, occluson, stationary)')
    parser.add_argument('--distance', action = 'store_true', default = False, help = 'Learn Distance Fields instead of lines')
    parser.add_argument('--NWS', action='store_true', help='Use NWS Data instead')
    parser.add_argument('--verbose', action='store_true', help='Print Current Step all the time')
    parser.add_argument('--normType', type = int, default = 0, help = 'normalization type used for the data:\n 0: min-max \n 1: mean-var \n 2: min-max per pixel \n 3: mean-var per pixel')
    parser.add_argument('--trainType', type = int, default = 0, help = 'defines which regions to use: 0: both \n 1: DWD NA \n 2: NWS hires')
    parser.add_argument('--IOU', action='store_true', help = 'use IOU error')
    parser.add_argument('--weight', type = float, default = 0.8, help = 'weight used for IOU error between detection and classification')
    

if __name__ == "__main__":
    BeginMultiprocessing(ParseArguments, runTraining)
