import numpy as np
from skimage import measure, morphology

# Returns filtered fronts excluding a border region
def filterFronts(image, border):
    threshold = 0.45
    minlen = 2
    filteredImage = np.zeros_like(image)
    numChannels = image.shape[-1]
    assert(numChannels > 1)
    for channel in range(numChannels):
        thinImg = image[:,border:-border, border:-border,channel]>threshold
        labeledImage = morphology.binary_dilation(thinImg, selem = np.ones((1,3,3)))
        #labeledImage = morphology.skeletonize(labeledImage)
        labeledImage = measure.label(labeledImage, background = 0)
        labeledImage *= thinImg
        numLabels = np.max(labeledImage)
        for pidx in range(1,numLabels+1):
            singleLabel = (labeledImage == pidx)*1
            points = np.nonzero(singleLabel)
            if len(points[0]) < minlen:
                labeledImage[points] = 0
        filteredImage[:,border:-border,border:-border,channel] = (labeledImage>0)*1
    return filteredImage