from pyhdf.SD import SD,SDC
import os
import numpy as np
from skimage.io import imsave


def getWarpMask(filename):
    file = SD(os.path.realpath(filename), SDC.READ)
    datasets = file.datasets()
    latmask = file.select('Latitude').get().astype(np.float64)
    lonmask = file.select('Longitude').get().astype(np.float64)
    sds_obj = file.select('cloud_top_temperature_1km')
    print("hello")
    print(sds_obj.attributes().items())
    #exit(1)
    return np.array([latmask, lonmask])