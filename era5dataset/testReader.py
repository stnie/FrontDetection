from ERA5Reader.readNetCDF import *
from ERA5Reader.util.readHDF4 import getWarpMask
import argparse
from skimage.io import imsave
import time

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='CDF_READER')
    parser.add_argument('--file', type = str, help='path no netcdf4-classic file')
    parser.add_argument('--vars', nargs="+" , type = str, default = None, help='path to folder containing data')
    parser.add_argument('--latrange', nargs=2, type = int, default = None, help='path to folder containing label')
    parser.add_argument('--lonrange', nargs=2, type = int, default = None, help='name of the output')
    parser.add_argument('--warpmask', type = str, default = None, help='path to a NASA hdf4 File containing a warpmask')

    args = parser.parse_args()

    myvars = args.vars
    if myvars is None:
        rootgrp = h5py.File(os.path.realpath(args.file), "r")
        myvars = sorted(list(np.asarray(rootgrp)))
        rootgrp.close()

    if(not args.warpmask is None):
        warpmask = getWarpMask(args.warpmask)
    else:
        warpmask = None
        
    latrange = args.latrange
    lonrange = args.lonrange

    if(latrange is None):
        latrange = (90, -90.01)
    if(lonrange is None):
        lonrange = (-180, 180)

    print("Extracting the following variables from file", args.file)
    print("Lat {} to {}, Lon {} to {}".format(*latrange, *lonrange))
    print(myvars)

    iters = 1
    print("Begin Timing different versions with {} Iterations each".format(iters))

    begin = time.time()
    for i in range(iters):
        pass
        #READ ETH Algorithm
        #img = extractImageFromCDFtmp(os.path.realpath(args.file), myvars, mlatrange,mlonrange,np.arange(105,138,2))
        #rootgrp = h5py.File(os.path.realpath(args.file), "r")
        rootgrp = Dataset(os.path.realpath(args.file), "r", format="NETCDF4", parallel=False)   
        img = rootgrp['FRONT'][0] 
        # roll along x axis (ETH uses 0 tp 360, we use -180 to 180)
        img = np.roll(img, 360)
        print(img.shape)
        imsave("ETH.png", img[0])
        rootgrp.close()
        exit(1)
    end = time.time()
    #print(img.shape)
    print(end-begin, "seconds elapsed")

    myReader = CDFReader(normType = None)
    myCDFReader = CDFReader(1, normType = None)
    begin = time.time()
    for i in range(iters):
        img3 = myReader.read(os.path.realpath(args.file), myvars, latrange, lonrange, np.arange(105,138,4), warpmask=warpmask)
    end = time.time()
    imsave("extractedEPT.png", img3[8])
    print(img3.shape)
    print(end-begin, "seconds elapsed read HDF5")
    print("warp error:")
    #print("longitude:", np.linalg.norm(img3[-2]-warpmask[1]))
    #print("latitude:", np.linalg.norm(img3[-1]-warpmask[0]))
    
    begin = time.time()
    for i in range(iters):
        img2 = myCDFReader.read(os.path.realpath(args.file), myvars, latrange, lonrange, np.arange(105,138,4), warpmask=warpmask)
    end = time.time()
    print(img2.shape)
    print("Difference Read HDF5 vs read CDF", np.linalg.norm(img3-img2))
    print(end-begin, "seconds elapsed read CDF")

    print("Check differences of my Reader to basic CDF Reader (only for basic variables)")
    for channel in range(len(myvars[:-2])):
        for level in range(9):
#        print(np.linalg.norm(img[i]-img2[i])/(40*80*4*4*6))
            diff = 0#np.linalg.norm(img[channel,level]-img3[channel*17+level])/(40*80*4*4*6)
            if diff > 0: 
               print(diff, myvars[channel], level)
#        print(np.linalg.norm(img[i]-img4[i])/(40*80*4*4*6))
    print("Save images")
    for idx, var in enumerate(myvars):
        for layer in range(1):
            if(not warpmask is None):
                img[idx,layer,((90-warpmask[0])*4).astype(int),((warpmask[1]-180)*4).astype(int)] += 0.5
            #imsave("testimgs/"+var+str(layer)+".png", (img[idx,layer,:,:]+1) /2)
            imsave("testimgs/"+var+str(layer)+"_normed.png", (img3[idx*9+layer,:,:]+1))
            imsave("testimgs/"+var+str(layer)+"_new.png", (img2[idx*9+layer,:,:]+1))
            pass
    exit(1)
    