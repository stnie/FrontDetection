
from ERA5Reader.readNetCDF import *
import argparse
from skimage.io import imsave

if __name__ ==  "__main__":


    parser = argparse.ArgumentParser(description='CDF_READER')
    parser.add_argument('--file', type = str, help='path no netcdf4-classic file')
    parser.add_argument('--vars', nargs="+" , type = str, default = None, help='path to folder containing data')
    parser.add_argument('--latrange', nargs=2, type = int, default = None, help='path to folder containing label')
    parser.add_argument('--lonrange', nargs=2, type = int, default = None, help='name of the output')
    parser.add_argument('--image', action = 'store_true', help= 'only create images, no comparison')

    args = parser.parse_args()

    myReader = CDFReader(normType = None)
    image = myReader.read(args.file, args.vars, args.latrange, args.lonrange)

    imsave("test.png", image[0])