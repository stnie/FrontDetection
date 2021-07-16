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

from FrontPostProcessing import filterFronts


class CSIEvaluator():
    def __init__(self, args):
        self.border = args.border
        self.halfRes = args.halfRes
        self.maxDist = args.maxDist
        self.ETH = args.ETH
        self.NWS = args.NWS
        self.globalCSI = args.globalCSI
        if(self.halfRes):
            pixPerDeg = 2
            self.res = (-0.5, 0.5)
        else:
            pixPerDeg = 4
            self.res = (-0.25, 0.25)
        self.northBorder = self.border*pixPerDeg + 5*pixPerDeg
        self.southBorder = self.border*pixPerDeg
        self.eastBorder = self.border*pixPerDeg  + 5*pixPerDeg
        self.westBorder = self.border*pixPerDeg
        if(args.NWS):
            self.offset = (self.border+10+5 , self.border+40)
        else:
            self.offset = (self.border+10+5 , self.border+135)
        self.avgCSI = np.zeros((5,7))
        
    def evaluate(self, labels, predictions, _):
        # downsample if necessary
        if(not self.ETH and self.halfRes):
            predictions = predictions.permute(0,3,1,2)
            predictions = torch.nn.functional.max_pool2d(predictions, kernel_size = 2)
            predictions = predictions.permute(0,2,3,1)
            labels = labels.permute(0,3,1,2)
            labels = torch.nn.functional.max_pool2d(labels, kernel_size = 2)
            labels = labels.permute(0,2,3,1)
        
        # extract the desired region
        predictions = predictions[:, self.northBorder:-self.southBorder,self.westBorder:-self.eastBorder].numpy()
        labels = labels[:, self.northBorder:-self.southBorder,self.westBorder:-self.eastBorder].numpy()
        # evaluate
        if(self.globalCSI):
            self.avgCSI[0] += self.getCriticalSuccessAgainstWholeInKM(np.sum(labels, axis=-1), predictions[:,:,:,0], self.res, self.offset, self.maxDist)
        else:
            self.avgCSI[0] += self.getCriticalSuccessInKM(np.sum(labels, axis=-1), predictions[:,:,:,0],self.res, self.offset, self.maxDist)
        if(not self.ETH):
            for chnl in range(labels.shape[-1]):
                if(self.globalCSI):    
                    self.avgCSI[chnl+1] += self.getCriticalSuccessAgainstWholeInKM(labels[:,:,:,chnl], predictions[:,:,:,chnl+1], self.res, self.offset, self.maxDist)
                else:
                    self.avgCSI[chnl+1] += self.getCriticalSuccessInKM(labels[:,:,:,chnl], predictions[:,:,:,chnl+1],self.res, self.offset, self.maxDist)

    def finish(self):
        globalPOD = self.avgCSI[:,3]/self.avgCSI[:,4]
        globalSR = self.avgCSI[:,5]/self.avgCSI[:,6]
        globalCSI = 1/(1/globalPOD + 1/globalSR - 1)
        if(args.NWS):
            totalPOD = np.sum(self.avgCSI[1:3,3])/np.sum(self.avgCSI[1:3,4])
            totalSR = np.sum(self.avgCSI[1:3,5])/np.sum(self.avgCSI[1:3,6])
            totalCSI =  1/(1/totalPOD + 1/totalSR - 1)
        else:
            print("DWD Labels has no stationary front, so we do not include them here!")
            totalPOD = np.sum(self.avgCSI[1:-1,3])/np.sum(self.avgCSI[1:-1,4])
            totalSR = np.sum(self.avgCSI[1:-1,5])/np.sum(self.avgCSI[1:-1,6])
            totalCSI =  1/(1/totalPOD + 1/totalSR - 1)
        totalErr = np.array([totalCSI,totalPOD,totalSR])
        globalErr = np.array([globalCSI, globalPOD, globalSR]).transpose()

        print("Global Count in label is \n{}".format(self.avgCSI[:, 4]))
        print("Global Count in prediction is \n{}".format(self.avgCSI[:, 6]))

        filename = os.path.join(name, "{}_{}_maxDist_{}.txt".format("global" if self.globalCSI else "local", "NWS" if self.NWS else "DWD", self.maxDist))
        if(not os.path.isdir(name)):
            os.mkdir(name)
        with open(filename, "w") as f:    
            print("global", self.globalCSI, "NWS Region", self.NWS, "maxDist", self.maxDist, file = f)
            print("Global CSI is \n{}".format(globalErr), file = f)
            print("Total CSI is \n{}".format(totalErr), file = f)
        print("global", self.globalCSI, "NWS Region", self.NWS, "maxDist", self.maxDist)
        print("Global CSI is \n{}".format(globalErr))
        print("Total CSI is \n{}".format(totalErr))
    
    def getCriticalSuccessInKM(self, image, prediction, res = (-0.25, 0.25), offset = (0,0), maxDist = 100):
        offarray = np.array([offset[0]/np.abs(res[0]), offset[1]/res[1]])

        # label all distinct components in both images
        mypred = morphology.binary_dilation(prediction[0,:,:]>0, selem = np.ones((3,3)))
        mythinpred = prediction[0,:,:]>0
        predLabel = measure.label(mypred, background = 0)*mythinpred

        myimg = morphology.binary_dilation(image[0,:,:]>0, selem = np.ones((3,3)))
        mythinimg = image[0,:,:]>0
        imageLabel = measure.label(myimg, background = 0)*mythinimg
        
        numPredLabels = np.max(predLabel)
        numImageLabels = np.max(imageLabel)

        numValidPredLabels = numPredLabels
        numValidImageLabels = numImageLabels

        SR = 1000*np.ones((numValidPredLabels, numValidImageLabels))
        POD = 1000*np.ones((numValidImageLabels, numValidPredLabels))

        for pidx in range(1,numValidPredLabels+1):
            onlyPredLabel = (predLabel == pidx)*(2*maxDist)
            labelPoints = np.nonzero(onlyPredLabel)
            labelPointsArr = np.array(labelPoints).transpose()+offarray
            for iidx in range(1,numValidImageLabels+1):
                onlyImageLabel = (imageLabel == iidx)*(2*maxDist)
                imagePoints = np.nonzero(onlyImageLabel)
                imagePointsArr = np.array(imagePoints).transpose()+offarray
                if(len(imagePointsArr)>0):
                    # for each predicted point get the minimum distance in km to the label
                    for p in range(len(labelPoints[0])):
                        onlyPredLabel[labelPoints[0][p],labelPoints[1][p]] = self.getDistanceOnEarth(labelPointsArr[p:p+1], imagePointsArr, res)
                medianDistance = np.median((onlyPredLabel)[labelPoints])
                SR[pidx-1, iidx-1] = min(SR[pidx-1, iidx-1], medianDistance)
                if(len(labelPointsArr)>0):
                    # for each predicted point get the minimum distance in km to the label
                    for p in range(len(imagePoints[0])):
                        onlyImageLabel[imagePoints[0][p],imagePoints[1][p]] = self.getDistanceOnEarth(imagePointsArr[p:p+1], labelPointsArr, res)

                medianDistance = np.median((onlyImageLabel)[imagePoints])
                POD[iidx-1, pidx-1] = min(POD[iidx-1, pidx-1], medianDistance)
        if(numPredLabels == 0 and numImageLabels == 0):
            return np.array([0,0,0, 0, 0, 0 ,0])
        elif(numPredLabels == 0 and numImageLabels != 0):
            return np.array([0,0,0, 0, numImageLabels, 0, 0])
        elif(numPredLabels != 0 and numImageLabels == 0):
            return np.array([0,0,0, 0, 0, 0, numPredLabels])
        bestPOD = np.min(POD, axis=1)
        bestSR = np.min(SR, axis = 1)
        
        myPOD = np.sum((bestPOD<maxDist)*1)/numImageLabels
        mySR = np.sum((bestSR<maxDist)*1)/numPredLabels
        myCSI = 0
        if(myPOD > 0 and mySR > 0):
            myCSI = 1/(1/myPOD + 1/mySR -1)
        return np.array([myCSI, myPOD, mySR, np.sum((bestPOD<=maxDist)*1), numImageLabels, np.sum((bestSR<=maxDist)*1), numPredLabels])

    def getCriticalSuccessAgainstWholeInKM(self, image, prediction, res = (-0.25, 0.25), offset = (0,0), maxDist=100):
        offarray = np.array([offset[0]/np.abs(res[0]), offset[1]/res[1]])
        # label all distinct components in both images
        mypred = morphology.binary_dilation(prediction[0,:,:]>0, selem = np.ones((3,3)))
        mythinpred = prediction[0,:,:]>0
        predLabel = measure.label(mypred, background = 0)*mythinpred

        myimg = morphology.binary_dilation(image[0,:,:]>0, selem = np.ones((3,3)))
        mythinimg = image[0,:,:]>0
        imageLabel = measure.label(myimg, background = 0)*mythinimg

        numPredLabels = np.max(predLabel)
        numImageLabels = np.max(imageLabel)

        numValidPredLabels = numPredLabels
        numValidImageLabels = numImageLabels
        
        SR = (2*maxDist)*np.ones((numValidPredLabels))
        POD = (2*maxDist)*np.ones((numValidImageLabels))
        imagePoints = np.nonzero(image[0])
        imagePointsArr = np.array(imagePoints).transpose()+offarray
        for pidx in range(1,numValidPredLabels+1):
            onlyPredLabel = (predLabel == pidx)*(2*maxDist)
            points = np.nonzero(onlyPredLabel)
            pointsArr = np.array(points).transpose()+offarray
            if(len(imagePointsArr)>0):
                for p in range(len(points[0])):
                    onlyPredLabel[points[0][p],points[1][p]] = self.getDistanceOnEarth(pointsArr[p:p+1], imagePointsArr, res)
            medianDistance = np.median((onlyPredLabel)[points])
            SR[pidx-1] = medianDistance
        
        labelPoints = np.nonzero(mythinpred)
        labelPointsArr = np.array(labelPoints).transpose()+offarray
        for iidx in range(1,numValidImageLabels+1):
            onlyImageLabel = (imageLabel == iidx)*(2*maxDist)
            points = np.nonzero(onlyImageLabel)
            pointsArr = np.array(points).transpose()+offarray
            if(len(labelPointsArr)>0):
                for p in range(len(points[0])):
                    onlyImageLabel[points[0][p],points[1][p]] = self.getDistanceOnEarth(pointsArr[p:p+1], labelPointsArr, res)
            medianDistance = np.median((onlyImageLabel)[points])
            POD[iidx-1] = medianDistance

        if(numPredLabels == 0 and numImageLabels == 0):
            return np.array([1,0,0, 0, 0, 0 ,0])
        elif(numPredLabels == 0 and numImageLabels != 0):
            return np.array([0,-1,-1, 0, numImageLabels, 0, 0])
        elif(numPredLabels != 0 and numImageLabels == 0):
            return np.array([0,-1,-1, 0, 0, 0, numPredLabels])
        bestPOD = POD
        bestSR = SR
    
        myPOD = np.sum((bestPOD<=maxDist)*1)/numImageLabels
        mySR = np.sum((bestSR<=maxDist)*1)/numPredLabels
        myCSI = 0
        if(myPOD > 0 and mySR > 0):
            myCSI = 1/(1/myPOD + 1/mySR -1)

        return np.array([myCSI, myPOD, mySR, np.sum((bestPOD<=maxDist)*1), numImageLabels, np.sum((bestSR<=maxDist)*1), numPredLabels])


    def getDistanceOnEarth(self, c1,c2, res):
        # earth radius in km
        latRes = 180/np.abs(res[0])+1
        lonRes = 360/np.abs(res[1])
        radius = 6371
        rad1 = self.pixelToAngle(c1, latRes, lonRes, res[0], res[1])
        rad2 = self.pixelToAngle(c2, latRes, lonRes, res[0], res[1])

        return self.getOnSphereDistance(rad1,rad2, radius)

    def pixelToAngle(self, p, latRes, lonRes, latStep, lonStep):
        lat = (p[:,0]-latRes//2)*latStep
        lon = (p[:,1]-lonRes/2)*lonStep
        latr = 2*np.pi*lat/360.0
        lonr = 2*np.pi*lon/360.0
        return np.array([latr, lonr])

    def getOnSphereDistance(self, c1, c2, radius):
        lat1, lon1 = c1
        lat2, lon2 = c2
        centralAngle = (np.arccos(np.clip(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2),-1,1)))
        d = radius*centralAngle
        return np.min(d)



class ClimatologyEvaluator():
    def __init__(self, args):
        latRes = 360 if args.halfRes or args.ETH else 720
        lonRes = 720 if args.halfRes or args.ETH else 1440

        self.ETH = args.ETH
        self.totalImage = torch.zeros((latRes, lonRes, args.classes))
        self.args = args
    def evaluate(self, label, fronts, filename):
        if(not self.ETH):
            # decrease the resolution
            fronts = fronts.permute(0,3,1,2)
            fronts = torch.nn.functional.max_pool2d(fronts, kernel_size=2, stride=2,padding = 0)
            fronts = fronts.permute(0,2,3,1)
        self.totalImage += fronts[0]

    def finish(self):
        name = os.path.join("Climatologies",self.args.outname)
        if(not os.path.isdir(name)):
            os.mkdir(name)
        
        self.totalImage = self.totalImage.cpu().numpy()
        typeToString = ["front", "warm","cold", "occ","stnry"]
        for frontalType in range(self.totalImage.shape[-1]):
            self.saveClimatology(self.totalImage[:,:,frontalType], num_samples, name, typeToString[frontalType])
    
    def saveClimatology(self, climatology, num_samples, name, typen):
        imsave(name+"/"+typen+"climatology.png", climatology/(num_samples))
        imsave(name+"/"+typen+"climatology_unnormalized.png", climatology)
        climatology = climatology.astype(np.float32)
        climatology.tofile(name+"/"+typen+"climatology.bin")

class DrawImageEvaluator():
    def __init__(self, args):
        self.outname = args.outname
    def evaluate(self, label, fronts, name):
        outfold = os.path.join("OutputImages", self.outname)
        if(not os.path.isdir(outfold)):
            os.mkdir(outfold)
        filename = os.path.splitext(name[0])[0]
        # save the label
        # switch channels for usual output colors
        outlabel = torch.zeros_like(label)
        # red -> warm
        outlabel[0,:,:,0] = label[0,:,:,0]
        # green -> stationary
        outlabel[0,:,:,1] = label[0,:,:,3]
        # blue -> cold
        outlabel[0,:,:,2] = label[0,:,:,1]
        # pink -> occlusion
        outlabel[0,:,:,0] = (outlabel[0,:,:,0]<=label[0,:,:,2])*label[0,:,:,2] + (outlabel[0,:,:,0] > label[0,:,:,2])*outlabel[0,:,:,0]
        outlabel[0,:,:,2] = (outlabel[0,:,:,2]<=label[0,:,:,2])*label[0,:,:,2] + (outlabel[0,:,:,2] > label[0,:,:,2])*outlabel[0,:,:,2]

        imsave(os.path.join(outfold, filename+"_label.png"), (outlabel.cpu().numpy()[0,20:-20,20:-20,:-1]*255).astype(np.uint8))
        # save the prediction
        #switch channels for usual output colors
        outpred = torch.zeros_like(label)
        # red -> warm
        outpred[0,:,:,0] = fronts[0,:,:,1]
        # green -> stationary
        outpred[0,:,:,1] = fronts[0,:,:,4]
        # blue -> cold
        outpred[0,:,:,2] = fronts[0,:,:,2]
        # pink -> occlusion
        outpred[0,:,:,0] = (outpred[0,:,:,0]<=fronts[0,:,:,3])*fronts[0,:,:,3] + (outpred[0,:,:,0] > fronts[0,:,:,3])*outpred[0,:,:,0]

        outpred[0,:,:,2] = (outpred[0,:,:,2]<=fronts[0,:,:,3])*fronts[0,:,:,3] + (outpred[0,:,:,2] > fronts[0,:,:,3])*outpred[0,:,:,2]

        imsave(os.path.join(outfold, filename+"_prediction.png"), (outpred.cpu().numpy()[0,20:-20,20:-20,:-1]*255).astype(np.uint8))
        
        outdiff = torch.zeros((label.shape[1], label.shape[2], 3))
        outdiff[:,:,0] = torch.sum(label[0,:,:,:], dim = -1).cpu()
        outdiff[:,:,1] = fronts[0,:,:,0].cpu()
        imsave(os.path.join(outfold, filename+"_diff.png"), (outdiff[20:-20,20:-20].numpy()*255).astype(np.uint8))


    def finish(self):
        print("Done") 

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
    parser.add_argument('--distance', action = 'store_true', help = 'Learn Distance Fields instead of lines')
    parser.add_argument('--normType', type = int, default = 0, help = 'How to normalize the data: 0 min-max, 1 mean-var, 2/3 the same but per pixel')
    parser.add_argument('--labelGroupingList', type = str, default = None, help = 'Comma separated list of label groups \n possible fields are w c o s (warm, cold, occluson, stationary)')
    parser.add_argument('--ETH', action = 'store_true', help = 'Compare against an ETH result instead of net')
    parser.add_argument('--show-error', action = 'store_true', help = 'show the inividual error values during inference')
    parser.add_argument('--fromFile', type = str, default = None, help = 'show the inividual error values during inference')
    parser.add_argument('--drawImages', action = 'store_true', help = 'draw results of each iteration')
    parser.add_argument('--elastic', action = 'store_true', help = 'use elastic fit loss')
    parser.add_argument('--CSI', action = 'store_true')
    parser.add_argument('--deeplab', action = 'store_true')
    parser.add_argument('--maxDist', type = int, default = 250, help = 'maxDist for CSI in km')
    parser.add_argument('--globalCSI', action='store_true', default = False, help = 'calculate CSI by matching each front against the whole set of predictions. Else each front is only matched against another front. ')
    parser.add_argument('--halfRes', action='store_true', help = 'evaluate on half resolution of ERA5')
    parser.add_argument('--border', type = int, default = 5, help = "A Border in degree which is not evaluated")
    parser.add_argument('--climatology', action='store_true', default = False, help = 'Create a Climatology instead of CSI calculation')
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
    if(args.ETH):
        args.halfRes = True
    if(args.climatology):
        args.fullsize = True
    # ETH extracts data at half res! The Network does not! 
    stepsize = 0.5 if args.ETH else 0.25
    if(args.fullsize):
        cropsize = (720, 1440)
        mapTypes = {"NA": ("NA", (90,-89.75), (-180,180), (-stepsize, stepsize), None) }
        if(args.NWS):
            mapTypes = {"hires": ("hires", (90, -89.75), (-180, 180), (-stepsize,stepsize), None) }
    elif(not args.fullsize and not args.halfRes):
        #cropsize = (200,360)
        #mapTypes = {"NA": ("NA", (80,30.25), (-45,45), (-stepsize, stepsize), None)}
        #if(args.NWS):
        #    cropsize = (200, 360) 
        #    mapTypes = {"hires": ("hires", (80, 30.25), (-140, -50), (-stepsize,stepsize), None) }
        cropsize = (184,360)
        mapTypes = {"NA": ("NA", (76,30.25), (-50,40), (-stepsize, stepsize), None)}
        if(args.NWS):
            cropsize = (184, 344) 
            mapTypes = {"hires": ("hires", (76, 30.25), (-141, -55), (-stepsize,stepsize), None) }
    # Comparison against ETH only uses midlatitudes
    elif(args.halfRes and not args.fullsize):
        cropsize = (34*4,360)
        mapTypes = {"NA": ("NA", (64,30.25), (-50,40), (-stepsize, stepsize), None)}
        if(args.NWS):
            cropsize = (34*4, 344) 
            mapTypes = {"hires": ("hires", (64, 30.25), (-141, -55), (-stepsize,stepsize), None) }

    # ETH uses half Res before
    if(args.ETH):
        cropsize=(cropsize[0]//2,cropsize[1]//2)
    
    myLevelRange = np.arange(105,138,4)

    myTransform = (None, None)
    labelThickness = 1
    labelTrans = (0,0)

    labelGroupingList = args.labelGroupingList
    myLineGenerator = extractStackedPolyLinesInRangeAsSignedDistance(labelGroupingList, labelThickness, labelTrans)
    if(args.elastic):
        myLineGenerator = extractCoordsInRange(labelGroupingList)
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
    if(ETH):
        myEraExtractor = ETHEraExtractor()
    # Create Dataset
    data_set = WeatherFrontDataset(data_dir=data_fold, label_dir=label_fold, mapTypes = mapTypes, levelRange = myLevelRange, transform=myTransform, outSize=cropsize, labelThickness= labelThickness, label_extractor = myLabelExtractor, era_extractor = myEraExtractor, asCoords = args.elastic, has_subfolds = (not (args.ETH),False), removePrefix = 1+(not args.ETH)*7)
    return data_set

def setupDataLoader(data_set, args):
    # Create DataLoader 
    sampler = SequentialSampler(data_set)
    numWorkers = 8
    if(args.ETH):
        numWorkers = 0
    loader = DataLoader(data_set, shuffle=False, 
    batch_size = 1, sampler = sampler, pin_memory = True, 
    collate_fn = collate_wrapper(args.stacked, args.elastic, 0), num_workers = numWorkers)
    return loader

def inferResults(model, inputs, args):
    if(args.ETH):
        outputs = inputs.permute(0,2,3,1)
        smoutputs = inputs.permute(0,2,3,1)
    else:
        outputs = model(inputs)
        if(args.deeplab):
            outputs = outputs["out"]
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
        smoutputs = filterFronts(smoutputs.cpu().numpy(), args.border*4)
    return outputs, smoutputs

def performInference(model, loader, num_samples, evaluator, parOpt, args):
    for idx, data in enumerate(tqdm(loader, desc ='eval'), 0):
        if(idx == num_samples):
            break
        inputs, labels, filename = data
        inputs = inputs.to(device = parOpt.device, non_blocking=False)
        # Create Results
        outputs, smoutputs = inferResults(model, inputs, args)
        if(args.climatology):
            if(args.ETH):
                # ETH does no additional filtering of the results
                evaluator.evaluate(None, smoutputs.cpu(), filename)
            else:
                # The net does perform some filterin steps first
                evaluator.evaluate(None, torch.from_numpy(smoutputs), filename)
        else:
            if(not args.elastic):
                labels = labels.to(device = parOpt.device, non_blocking=False)
            pixPerDeg = 2 if args.ETH else 4
            labels = filterFronts(labels.cpu().numpy(), args.border*pixPerDeg)
            if(args.ETH):
                evaluator.evaluate(torch.from_numpy(labels), smoutputs.cpu(), filename)
            else:
                evaluator.evaluate(torch.from_numpy(labels), torch.from_numpy(smoutputs), filename)
    evaluator.finish()

def setupModel(args, parOpt):
    model = None
    if(not args.ETH):
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

    name = os.path.join("Predictions",args.outname)
    
    ETH = args.ETH

    args.stacked = True
    data_set = setupDataset(args)    
    loader = setupDataLoader(data_set, args)
    

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
    
    # Basic: CSI Evaluation
    evaluator = CSIEvaluator(args)
    # Alternative: Create A climatology of a year
    if(args.climatology):
        evaluator = ClimatologyEvaluator(args)
    if(args.drawImages):
        evaluator = DrawImageEvaluator(args)

    num_samples = len(loader)
    if(args.num_samples != -1):
        num_samples = args.num_samples

    print("Evaluating {} Data files".format(num_samples))
    with torch.no_grad():
        avg_error = performInference(model, loader, num_samples, evaluator, parOpt, args)
        
