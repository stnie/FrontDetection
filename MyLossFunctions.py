import torch
import numpy as np
from skimage.draw import line
import CModules.viterby.viterby as vd

# This class gets a multichannel network output (without activation function)
# channel 0 is considered the background
class IoUWithFitRaw():
    def __init__(self, weight = 0.8, interChannelWeight = 1, maxDist = 3, sigma = 3, deletion_error = -10000, channel_change = False, border = 0):
        self.weight = weight
        self.interChannelWeight = interChannelWeight
        self.maxDist = maxDist
        self.sigma = sigma
        self.border = border
        self.deletion_error = deletion_error
        self.channel_change = channel_change
        self.transform_penalty = torch.Tensor([
            [1    , 0.2, 0.2, 0.2],
            [0.2 , 1   , 0.2, 0.2],
            [0.2 , 0.2, 1   , 0.1],
            [0.0 , 0.0, 0.0, 1  ]]).cuda()
        self.softDeletion = True
        self.detectionQuality = torch.Tensor([0.0]).cuda()
        self.ignoreZeroLoss = True
        self.wholeRegion = True

    def __call__(self, pred, labcoords, masks = None):
        # prediction is multichanneled raw network output
        # resample the output into a probability distribution
        local_pred = torch.nn.functional.softmax(pred, dim = 3)
        # NoObject holds the probabilty that no front is detected
        # As a result 1-NoObject is the prediction probability that any kind of front is present
        NoObject = local_pred[:,:,:,0]
        AnyObject = 1-NoObject
        # Channel Objects hold the individual probabilty for each channel to be detected
        ChannelObjects = local_pred[:,:,:,1:]
        # Returns the label, with channel propositions
        if(self.wholeRegion):
            lab = self.getBestFit(AnyObject.detach(), ChannelObjects.detach(), labcoords)
        else:
            lab = self.getRandomFit(AnyObject.detach(), ChannelObjects.detach(), labcoords)
        

        # the binary label is just the sum over all channels (label channels are disjoint sets)
        binaryLab = torch.sum(lab, dim = 3)

        # Remove all data which is not in the analysis region denoted by map
        if((masks is None)):
            # first calculate the binary loss
            binProd = torch.sum((AnyObject*binaryLab)[:,self.border:AnyObject.shape[1]-self.border,self.border:AnyObject.shape[2]-self.border], dim = (1,2))
            binPredSum = torch.sum((AnyObject*AnyObject)[:,self.border:AnyObject.shape[1]-self.border,self.border:AnyObject.shape[2]-self.border], dim = (1,2))
            binLabSum = torch.sum((binaryLab*binaryLab)[:,self.border:AnyObject.shape[1]-self.border,self.border:AnyObject.shape[2]-self.border], dim = (1,2))
            # Now get the Channel loss 
            chanProd = torch.sum((ChannelObjects*lab)[:,self.border:ChannelObjects.shape[1]-self.border,self.border:ChannelObjects.shape[2]-self.border], dim = (1,2))
            chanPredSum = torch.sum((ChannelObjects*ChannelObjects)[:,self.border:ChannelObjects.shape[1]-self.border,self.border:ChannelObjects.shape[2]-self.border], dim = (1,2))
            chanLabSum = torch.sum((lab*lab)[:,self.border:ChannelObjects.shape[1]-self.border,self.border:ChannelObjects.shape[2]-self.border], dim = (1,2))
        else:
            masks[:,-self.border:self.border:,-self.border:self.border] = 0
            points = [torch.nonzero(masks[x]) for x in range(AnyObject.shape[0])]

            binProd = torch.zeros((lab.shape[0]),requires_grad = True).cuda()
            binPredSum = torch.zeros((lab.shape[0]),requires_grad = True).cuda()
            binLabSum = torch.zeros((lab.shape[0]),requires_grad = True).cuda()
            for inst in range(lab.shape[0]):
                binProd[inst] += torch.sum((AnyObject*binaryLab)[inst,points[inst][:,0],points[inst][:,1]])
                binPredSum[inst] += torch.sum((AnyObject*AnyObject)[inst, points[inst][:,0], points[inst][:,1]])
                binLabSum[inst] += torch.sum((binaryLab*binaryLab)[inst, points[inst][:,0], points[inst][:,1]])
            
            chanProd = torch.zeros((lab.shape[0],lab.shape[-1]),requires_grad = True).cuda()
            chanPredSum = torch.zeros((lab.shape[0],lab.shape[-1]),requires_grad = True).cuda()
            chanLabSum = torch.zeros((lab.shape[0],lab.shape[-1]),requires_grad = True).cuda()
            for inst in range(lab.shape[0]):
                chanProd[inst] += torch.sum((ChannelObjects*lab)[inst,points[inst][:,0], points[inst][:,1]], dim = 0)
                chanPredSum[inst] += torch.sum((ChannelObjects*ChannelObjects)[inst,points[inst][:,0], points[inst][:,1]],dim = 0)
                chanLabSum[inst] += torch.sum((lab*lab)[inst,points[inst][:,0], points[inst][:,1]],dim = 0)


        binaryLoss = torch.mean(1-(binProd / (binPredSum+binLabSum - binProd)))
        if(self.ignoreZeroLoss):
            # Get only channels where the corresponding label is present
            # This will only return a gradient for samples that contain a label.
            nonZeroChannels = torch.nonzero(chanLabSum, as_tuple=False)
            counts = 1.0*torch.Tensor([torch.sum(nonZeroChannels[:,1] == x) for x in range(chanLabSum.shape[1])])

            # count the number of nonzero labels in this batch
            channelWeight = (chanLabSum.shape[0]/(counts)).cuda()
            channelWeight *= self.interChannelWeight
            # loss for each batch object and channel
            channelLoss = 1-(chanProd[nonZeroChannels[:,0],nonZeroChannels[:,1]] / (chanPredSum[nonZeroChannels[:,0],nonZeroChannels[:,1]]+chanLabSum[nonZeroChannels[:,0],nonZeroChannels[:,1]] - chanProd[nonZeroChannels[:,0],nonZeroChannels[:,1]]))
            # for each channel
            
            for x in range(chanLabSum.shape[1]):
                channelLoss[nonZeroChannels[:,1]==x] *= channelWeight[x]
        else:
            # Add small epsilon to the loss to counter 0 division cases. Still returns gradient for each sample
            epsilon = 0.00001
            channelLoss = 1-((chanProd+epsilon) / (chanPredSum+chanLabSum - chanProd+epsilon))
        # get the mean weighted by the per channel weights
        channelLoss = torch.sum(channelLoss)/(torch.sum(self.interChannelWeight)*pred.shape[0])
        return self.weight*binaryLoss + (1-self.weight)*channelLoss

    def getRandomFit(self, image, channelImage, coords):
        reconImage = torch.zeros_like(channelImage).cuda()
        # flattened image
        myImage = image.cpu().numpy()
        groups = np.random.randint(0, len(coords[0]), size = len(coords))
        for instance in range(len(coords)):
            group = groups[instance]
            for front in coords[instance][group]:
                reconstructedCoords = vd.fit_line(front, myImage[instance], self.maxDist, self.sigma, self.deletion_error, self.border)
                for pairIdx in range(reconstructedCoords.shape[0]-1):
                    myStart = reconstructedCoords[pairIdx]
                    myEnd = reconstructedCoords[pairIdx+1]
                    lineValue = 1
                    if(np.linalg.norm(myStart-myEnd) == 0):
                        continue
                    if(myStart[0] == -10000 or myEnd[0] == -10000):
                        lineValue = 0
                        if(self.softDeletion):
                            # Soft Deletion (set value to chance of random deletion)
                            # first pair is deletion candidate
                            if(myStart[0] == -10000):
                                myStart = front[pairIdx]
                            if(myEnd[0] == -10000):
                                myEnd = front[pairIdx+1]
                            numPixel = np.abs(myEnd[:2]-myStart[:2]).max()*-self.deletion_error
                            # Calculate the chance that enough points were randomly not detected to be considered a deletion
                            # That value will be written instead
                            lineValue = torch.pow(1-self.detectionQuality,float(numPixel))

                        # shortcut for true deletion
                        if(lineValue == 0):
                            continue
                    rr, cc = line(reconstructedCoords[pairIdx,0],reconstructedCoords[pairIdx,1],reconstructedCoords[pairIdx+1,0],reconstructedCoords[pairIdx+1,1])
                    pos = np.where((rr>=0) & (rr<myImage.shape[1]) & (cc >= 0) & (cc < myImage.shape[2]))
                    rr = rr[pos]
                    cc = cc[pos]
                    targetGroup = group
                    if(self.channel_change and rr.shape[0] > 0):
                        # evaluate the loss function again, to determine the optimal color channel
                        channelScore = (1+self.transform_penalty[group])*channelImage[instance,rr,cc]-1
                        # the target stores the channel for this
                        # get optimal channel
                        targetGroup = torch.max(channelScore,1)[1]
                    
                    # Draw into the target channel
                    reconImage[instance, rr, cc, targetGroup] = lineValue
        return reconImage
    
    def getBestFit(self, image, channelImage, coords):
        reconImage = torch.zeros_like(channelImage).cuda()
        # flattened image
        myImage = image.cpu().numpy()
        for instance in range(len(coords)):
            for group in range(len(coords[instance])):
                assert(len(coords[instance]) == channelImage.shape[-1])
                for front in coords[instance][group]:
                    #front-=self.border
                    #score, pos, paths = getBestPath(front, image[instance], self.maxDist, useTorch = True)
                    #reconstructedCoords = getOptPath(paths, pos, front, self.maxDist).astype(np.int32)
                    reconstructedCoords = vd.fit_line(front, myImage[instance], self.maxDist, self.sigma, self.deletion_error, self.border)
                    for pairIdx in range(reconstructedCoords.shape[0]-1):
                        myStart = reconstructedCoords[pairIdx]
                        myEnd = reconstructedCoords[pairIdx+1]
                        lineValue = 1
                        if(np.linalg.norm(myStart-myEnd) == 0):
                            continue
                        if(myStart[0] == -10000 or myEnd[0] == -10000):
                            lineValue = 0
                            if(self.softDeletion):
                                # Soft Deletion (set value to chance of random deletion)
                                # first pair is deletion candidate
                                if(myStart[0] == -10000):
                                    myStart = front[pairIdx]
                                if(myEnd[0] == -10000):
                                    myEnd = front[pairIdx+1]
                                numPixel = np.abs(myEnd[:2]-myStart[:2]).max()*-self.deletion_error
                                # Calculate the chance that enough points were randomly not detected to be considered a deletion
                                # That value will be written instead
                                lineValue = torch.pow(1-self.detectionQuality,float(numPixel))

                            # shortcut for true deletion
                            if(lineValue == 0):
                                continue
                        rr, cc = line(reconstructedCoords[pairIdx,0],reconstructedCoords[pairIdx,1],reconstructedCoords[pairIdx+1,0],reconstructedCoords[pairIdx+1,1])
                        pos = np.where((rr>=0) & (rr<myImage.shape[1]) & (cc >= 0) & (cc < myImage.shape[2]))
                        rr = rr[pos]
                        cc = cc[pos]
                        targetGroup = group
                        if(self.channel_change and rr.shape[0] > 0):
                            # evaluate the loss function again, to determine the optimal color channel
                            channelScore = (1+self.transform_penalty[group])*channelImage[instance,rr,cc]-1
                            # the target stores the channel for this
                            # get optimal channel
                            targetGroup = torch.max(channelScore,1)[1]
                        
                        # Draw into the target channel
                        reconImage[instance, rr, cc, targetGroup] = lineValue
        return reconImage

if __name__ == "__main__":
    pass
