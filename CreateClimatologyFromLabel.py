import numpy as np
from skimage.io import imsave
import sys
import os
from IOModules.csbReader import *

if __name__ == "__main__":
    datafold = sys.argv[1]
    possibleYears = ["2016"]#["2016","2017","2018","2019"]
    possibleMonths = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    labelCropSize = (720//2, 1440//2)
    image = np.zeros((labelCropSize))
    labelThickness = 1
    labelTrans = (0,0)
    labelGrouping = "wco"
    myLineGenerator = extractStackedPolyLinesInRangeAsSignedDistance(labelGrouping, labelThickness, labelTrans)
    # elastic label is only the coordinate pairs
    myLabelExtractor = DefaultFrontLabelExtractor(myLineGenerator)


    for year in os.listdir(datafold):
        if(year in possibleYears):
            print(year, flush=True)
            yearpath = os.path.join(datafold, year)
            for month in os.listdir(yearpath):
                if(month in possibleMonths):
                    print("\t",month, flush = True)
                    monthpath = os.path.join(yearpath, month)
                    for filename in os.listdir(monthpath):
                        if("_03" in filename or "_09" in filename or "_15" in filename or "_21" in filename):
                            continue
                        image += myLabelExtractor(os.path.join(monthpath,filename), (90,-89.75), (-180,180), (-0.5,0.5), "hires")[:,:,0]
    outfold = os.path.join("Climatologies", sys.argv[2])
    if(not os.path.exists(outfold)):
        os.mkdir(outfold)
    outname = os.path.join("Climatologies", sys.argv[2], "climatology")
    imsave(outname+".png", image)
    image = image.astype(np.float32)
    image.tofile(outname+".bin")

