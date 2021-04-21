from FrontDataset import *
from skimage.io import imsave, imread
import argparse
import numpy as np


def drawExamples(dataset, indices = None, num_examples = 1, off = (0,0)):
    outfold = "img_checks"
    if(indices is None):
        indices = np.arange(num_examples)
    for index in indices:
        smpl = data_set.__getitem__(index)
        #transposed_sample = list(zip(*smpl))
        #data = torch.stack(transposed_sample[0],0)
        #lab = torch.stack(transposed_sample[1],0)
        data, lab, name = smpl
        print(data.shape, name)
        data = data.cpu().numpy()
        data =data.astype(np.float32)
        data.tofile(outfold+"/"+"all.bin")
        for channel in range(data.shape[0]):
            #imsave(outfold+"/"+"ex{}_{}_{}.png".format(index, channel, level), data[channel, level,:,:].numpy(), check_contrast=False)
            #data[channel, level, off[0]:lab.shape[0]+off[0],off[1]:lab.shape[1]+off[1]] = data[ channel, level, off[0]:lab.shape[0]+off[0],off[1]:lab.shape[1]+off[1]]+2*torch.max(data[channel,level,:,:])*(lab[:,:,0])
            imsave(outfold+"/"+"comb_ex{}.png".format(channel), data[channel], check_contrast=False)
            #break
        #imsave(outfold+"/"+"gt{}.png".format(index), lab[:,:,0].numpy().astype(np.uint8), check_contrast=False)



def setupDataset(data_location, label_location, variables):
    
    outsizes = (721, 1440)
    mapTypes = {"NA": ("NA", (90,-90.01), (-180,180), (-0.25,0.25)) }
    #outsizes = (160, 256)
    #mapTypes = {"NA": ("NA", (70,30), (-40,40), (-0.25,0.25)) }

    transforms = (None, None)

    myLevelRange = np.arange(105,138,4)
    # train data set
    data_fold = data_location
    label_fold = label_location

    labelThickness = 1
    labelGroupingList = 'wcos'

    myLabelExtractor = None

    myEraExtractor = DerivativeFlippingAwareEraExtractor(variables, fliprate = 0.0)
    myEraExtractor = ETHEraExtractor()

    data_set = WeatherFrontDataset(data_dir=data_fold, label_dir=label_fold, mapTypes=mapTypes, levelRange = myLevelRange, transform=transforms, outSize= outsizes, labelThickness = labelThickness, label_extractor = myLabelExtractor, era_extractor = myEraExtractor)

    print()
    print("Training Data...")
    print("Data-Location:", data_fold)
    print("Label-Location:", label_fold)
    print()

    return data_set



def generateDefaultDataset(type = "DWD"):


    myTypes = ["NA", "NT"]
     # NA to NA, NT to NT. A more generous range of lons and lats to contain most fronts of the images
    myLats = [(30, 80), (20, 90)]
    myLons = [(-60, 70), (-80, 100)]

    # NA to NA, NT to NT. A more conservative range of lons and lats to reduce points not present in the labeled data 
    myLatsSafe = [(30, 65), (30, 90)]    # in px [160 ,  240]
    myLonsSafe = [(-40, 40), (-80, 40)]  # in px [320 ,  480]

    resolution = [(0.25, 0.25), (0.25, 0.25)]

    myMapTypes = {"NA": ("NA",myLatsSafe[0], myLonsSafe[0], resolution[0])}

    myLevelRange = np.arange(105,138,4)

    cropSize = (128,256)
    lcropSize = (102,230)
    sizes = (cropSize, lcropSize)

    affine = RandomAffine(degrees = 10, translate = (0.05, 0.05), scale = (1,1), shear = (-10,10), resample = False, fillcolor = 0)
    crop = RandomCrop(cropSize, pad_if_needed=True)
    lcrop = RandomCrop(cropSize, pad_if_needed=True)
    fcrop = FixedCrop(lcropSize)
    erasor = RandomErasing()
    noise = GaussianNoise(1)
    myTransform = Compose([erasor])
    mylTransform = Compose([erasor, noise,fcrop])
    theTransform = (None, mylTransform)
    data_fold = 'data/test2/test_data'
    label_fold = 'data/test2/test_fronts'

    if(type == "NWS"):
        myLatsSafe = [(10, 60)]
        myLonsSafe = [(-140, -50)]
        resolution = [(0.25, 0.25)]
        myTypes = ["hires"]
        data_fold = 'data/testNWS/test_data'
        label_fold = 'data/testNWS/test_fronts'

    data_set = WeatherFrontDataset(data_dir=data_fold, label_dir=label_fold, mapTypes=myMapTypes, levelRange = myLevelRange, transform=theTransform, outSize=sizes)
    return data_set

if __name__ == "__main__":
    '''
    build a sample database and save data, label and the mapped images
    '''
    parser = argparse.ArgumentParser(description='FrontNet')
    parser.add_argument('--data', help='path to folder containing data')
    parser.add_argument('--label', type = str, default = None, help='path to folder containing label')
    parser.add_argument('--vars', nargs="+" , type = str, default = None, help='path to folder containing data')
    args = parser.parse_args()

    #vars = ['t', 'sp', 'delta_t_udir', 'delta_q_udir','delta_q_vdir', 'u_v_2pol','u_v_2abs', 'delta_t_vdir','q','v','u','w']
    vars = args.vars

    data_set = setupDataset(args.data, args.label, vars)
    num_samples = 1
    print(data_set)

  
    draw = True
    if(draw):
        # check to see if the fronts are drawn correctly
        drawExamples(data_set, np.array([random.randint(0,len(data_set)-1) for x in range(num_samples)]))
    else:
        for sample in range(num_samples):
            idx = random.randint(0,len(data_set)-1)
            smpl = data_set.__getitem__(idx)
            print("reading sample:",sample, "/", num_samples,"index:", idx,"/",len(data_set), ":", smpl[2])
    #lt.pend("all")
    #pr.print_stats(10)