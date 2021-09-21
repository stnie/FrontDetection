from netCDF4 import Dataset
import h5py
import os
import numpy as np
from .L137Levels import L137Calculator
from scipy.ndimage import map_coordinates , geometric_transform
#from .util.CModules.CalcVariables.example import ept
from multiprocessing.shared_memory import SharedMemory
# for upscaling of ETH 
from skimage.transform import pyramid_expand
from skimage.morphology import skeletonize


class ETHReader:
    def __init__(self):
        pass
    def read(self, filename, latrange, lonrange):
        rootgrp = Dataset(os.path.realpath(filename), "r", format="NETCDF4", parallel=False)
        img = rootgrp['FRONT'][0] 
        # upscale then delete the last row
        #imgups = pyramid_expand(img[0])[:,:]
        #img = np.reshape(imgups, (1,*imgups.shape))
        # Now img is in the same format as our results
        # Get The offsets for cropping
        mylonRange = (np.arange(lonrange[0], lonrange[1], 0.5)*2).astype(np.int32)
        mylatRange = (np.arange(latrange[0], latrange[1], -0.5)*2).astype(np.int32)
        # Adjust the range to the data format of ETH  (lat -89.5 -> 90, lon -180 -> 180 (after np roll))
        
        mylonRange += 180*2
        mylatRange -= 90*2+1
        img = img[:, mylatRange, :]
        img = img[:,:,mylonRange]
        rootgrp.close()
        #img = skeletonize(img>0)*img
        return img

class BinaryResultReader:
    def __init__(self):
        pass
    def read(self, filename, latrange, lonrange):
        data = np.fromfile(filename, dtype=np.bool).reshape(720,1440,5)
        mylonRange = (np.arange(lonrange[0], lonrange[1], 0.25)*4).astype(np.int32)
        mylatRange = (np.arange(latrange[0], latrange[1], -0.25)*4).astype(np.int32)
        mylonRange += 180*4
        mylatRange = 90*4 - mylatRange
        img = data[mylatRange]
        img = img[:,mylonRange]
        return img


class CDFReader:
    def __init__(self, filetype = 0, normType = None, sharedObj = None):
        # 0 -> h5py, 1 -> netCDF4
        self.filetype = filetype
        self.asHDF5 = filetype == 0
        #None -> No normalization, 0 -> min,max , 1 -> mean, var
        self.normalize_type = normType
        self.normalize = not self.normalize_type is None

        self.warp_func = np.vectorize(map_coordinates, signature='(b,m),(2,l,n)->(l,n)')
        if(sharedObj is None):
            self.bufferMasks = False
        else:
            self.bufferMasks = True
            self.cnt = 0
            self.lock = sharedObj[0]
            self.pathToMasks = sharedObj[1]

    def __repr__(self):
        myString = "CDFReader\n"
        myString += str(self.__dict__)
        return myString 

    def read(self, filename, variables = None, latrange = None, lonrange = None, levelrange = None, lat_step = None, lon_step = None, warpmask = None):
        # Open the file depending on the filetype
        if(self.asHDF5):
            rootgrp = h5py.File(os.path.realpath(filename), "r")
        else:
            rootgrp = Dataset(os.path.realpath(filename), "r", format="NETCDF4", parallel=False)
        
        # If no variables are given, we extract all variables in the dataset
        if(variables is None):
            variables = np.asarray(rootgrp)

        # Set the local values first
        local_latrange, local_lonrange, local_levelrange, local_warpmask = getLocalValues(rootgrp, latrange, lonrange, levelrange, lat_step, lon_step, warpmask)
       
        # Read the values from the file
        myImage = extractImageFromCDFh5pyChunkedSlim1dAfterNormGeneralDerivativeAndCache(rootgrp, variables, local_latrange, local_lonrange, local_levelrange, self.asHDF5)

        # Here we can already close the file before post processing
        rootgrp.close()
        # normalize the image if desired
        if(self.normalize):
            self.normalizeImageInPlace(myImage, variables, local_latrange, local_lonrange, local_levelrange, filename)
        
        # warp image if a warp mask is given
        performWarp = not local_warpmask is None
        if(performWarp):
            myImage = self.warp_func(myImage, local_warpmask)

        # currently not used
        reverse_y_axis = False
        reverse_x_axis = False
        if(reverse_y_axis):
            myImage = np.flip(myImage,axis=3)
        if(reverse_x_axis):
            myImage = np.flip(myImage,axis=4)

        # if reversal is necessary we need to return a copy rather than a view
        # this is however slower
        if(reverse_x_axis or reverse_y_axis):
            return myImage.copy()
        return myImage
        
    def normalizeImageInPlace(self, img, variables, latrange, lonrange, levelrange, name):
        idx = 0
        svalIdx = 0
        levels = len(levelrange)
        date = getDateFromName(name)
        for variable in variables:
            # remove cached identifier
            if(variable[-2:] == "_c"):
                variable = variable[:-2]
            
            # get index range
            idxRange = slice(idx*levels,(idx+1)*levels,1)
            idx+=1
            
            normalization_method = self.normalize_type
            # Latitude Longitude or distortion are fixed values and do not need
            # to be normalized by seasons. (as this would neutralize any difference)
            if(variable in ['latitude','longitude','kmPerLon','kmPerLat']):
                idxRange = -svalIdx-1
                idx-=1
                svalIdx += 1
                if(normalization_method > 1):
                    normalization_method -= 2
            # do not normalize a base variable
            if('base' in variable):
                normalization_method = -1
            
            # Normalization offsets
            if(normalization_method == 1):
                mean,var = getMeanVar(variable)
                img[idxRange] -= mean
                img[idxRange] *= 1/np.sqrt(var)
            elif(normalization_method == 0):
                minval,maxval = getValueRanges(variable)    
                valRange = (maxval-minval)        
                img[idxRange] -= (minval+maxval)/2
                img[idxRange] *= (2/valRange)
            elif(normalization_method == 2):
                # normalize the selected layers within the function (that way all shared memory access is capsuled within the function)
                self.getNormalizationMaskshm(variable, latrange, lonrange, levelrange, date, img[idxRange], 0)
            elif(normalization_method == 3):
                # normalize the selected layers within the function (that way all shared memory access is capsuled within the function)
                self.getNormalizationMaskshm(variable, latrange, lonrange, levelrange, date, img[idxRange], 1)
    

    def getNormalizationMask(self, variable, latrange, lonrange, levelrange, date, type):
        year,month,day,hour = date
        target_name = variable+"_m"+str(int(month))+"_h"+str(int(hour))+".bin"
        
        pathToMasks = self.pathToMasks
        if(type == 0):
            mask_file_string1 = os.path.join(pathToMasks,"statimgs","min",target_name)
            mask_file_string2 = os.path.join(pathToMasks,"statimgs","max",target_name)
        elif(type == 1):
            mask_file_string1 = os.path.join(pathToMasks,"statimgs","mean",target_name)
            mask_file_string2 = os.path.join(pathToMasks,"statimgs","var",target_name)
        if(self.bufferMasks):
            # if not already loaded, load the mask before returning it
            isLoaded = mask_file_string1 in self.maskLoaded and self.maskLoaded[mask_file_string1] and mask_file_string2 in self.maskLoaded and self.maskLoaded[mask_file_string2]
            if(not isLoaded):
                self.lock.acquire()
                isLoaded = mask_file_string1 in self.maskLoaded and self.maskLoaded[mask_file_string1] and mask_file_string2 in self.maskLoaded and self.maskLoaded[mask_file_string2]
                if(not isLoaded):
                    local_level_step = 1
                    if(levelrange.shape[0]>1):
                        local_level_step = levelrange[1]-levelrange[0]
                    ExtractionLevelRange = slice(levelrange[0],levelrange[-1]+1,local_level_step)
                    print("inserting:", mask_file_string1)
                    if(type == 0):
                        # load min and max and generate (min+max)/2 as well as the inverse 
                        m1 = np.reshape(np.fromfile(mask_file_string1), (17,721,1440))[ExtractionLevelRange]
                        m1 += np.reshape(np.fromfile(mask_file_string2), (17,721,1440))[ExtractionLevelRange]
                        m1 *= 0.5
                        m2 = np.reshape(np.fromfile(mask_file_string2), (17,721,1440))[ExtractionLevelRange]
                        m2 -= np.reshape(np.fromfile(mask_file_string1), (17,721,1440))[ExtractionLevelRange]
                        m2 = 2/m2
                    elif(type == 1):
                        # load mean
                        m1 = np.reshape(np.fromfile(mask_file_string1), (17,721,1440))[ExtractionLevelRange]
                        # load variance and save inverse root
                        m2 = np.reshape(np.fromfile(mask_file_string2), (17,721,1440))[ExtractionLevelRange]
                        m2 = 1/np.sqrt(m2)
                    self.mask[mask_file_string1] = m1
                    self.mask[mask_file_string2] = m2
                    self.maskLoaded[mask_file_string1] = True
                    self.maskLoaded[mask_file_string2] = True
                    print("size is now:", len(self.mask))
                self.lock.release()
            # get local offsets
            local_lat_step = 1
            local_lon_step = 1
            if(latrange.shape[0]>1):
                local_lat_step = latrange[1]-latrange[0]
            if(lonrange.shape[0]>1):
                local_lon_step = lonrange[1]-lonrange[0]
            extraction_location = (slice(0, len(levelrange), 1),slice(latrange[0],latrange[-1]+1,local_lat_step),slice(lonrange[0],lonrange[-1]+1,local_lon_step))
            mask1 = self.mask[mask_file_string1][extraction_location]
            mask2 = self.mask[mask_file_string2][extraction_location]
                
        else:
            # get local offsets
            local_lat_step = 1
            local_lon_step = 1
            local_level_step = 1
            if(levelrange.shape[0]>1):
                local_level_step = levelrange[1]-levelrange[0]
            if(latrange.shape[0]>1):
                local_lat_step = latrange[1]-latrange[0]
            if(lonrange.shape[0]>1):
                local_lon_step = lonrange[1]-lonrange[0]
            extraction_location = (slice(levelrange[0],levelrange[-1]+1,local_level_step),slice(latrange[0],latrange[-1]+1,local_lat_step),slice(lonrange[0],lonrange[-1]+1,local_lon_step))
            if(type == 0):
                # load min and max and generate (min+max)/2 as well as the inverse 
                mask1 = np.reshape(np.fromfile(mask_file_string1), (17,721,1440))[extraction_location]
                mask1 += np.reshape(np.fromfile(mask_file_string2), (17,721,1440))[extraction_location]
                mask1 *= 0.5
                mask2 = np.reshape(np.fromfile(mask_file_string2), (17,721,1440))[extraction_location]
                mask2 -= np.reshape(np.fromfile(mask_file_string1), (17,721,1440))[extraction_location]
                mask2 = 2/mask2
            elif(type == 1):
                # load mean
                mask1 = np.reshape(np.fromfile(mask_file_string1), (17,721,1440))[extraction_location]
                # load variance and save inverse root
                mask2 = np.reshape(np.fromfile(mask_file_string2), (17,721,1440))[extraction_location]
                mask2 = 1/np.sqrt(mask2)

        return mask1, mask2

    def getNormalizationMaskshm(self, variable, latrange, lonrange, levelrange, date, img, type):
        year,month,day,hour = date
        target_name = variable+"_m"+str(int(month))+"_h"+str(int(hour))+".bin"
        
        pathToMasks = self.pathToMasks#os.path.join("/home","stefan","Documents","Binary","Front-Detection","myTrainingScripts","era5dataset")
        if(type == 0):
            mask_file_string1 = os.path.join(pathToMasks,"statimgs","min",target_name)
            mask_file_string2 = os.path.join(pathToMasks,"statimgs","max",target_name)
            shm_string1 = "min"+month+hour+variable
            shm_string2 = "max"+month+hour+variable

        elif(type == 1):
            mask_file_string1 = os.path.join(pathToMasks,"statimgs","mean",target_name)
            mask_file_string2 = os.path.join(pathToMasks,"statimgs","var",target_name)
            shm_string1 = "mean"+month+hour+variable
            shm_string2 = "var"+month+hour+variable
        if(self.bufferMasks):
            # if not already loaded, load the mask before returning it
            try:
                shm1 = SharedMemory(shm_string1)
                shm2 = SharedMemory(shm_string2)
                isLoaded = True
            except:
                isLoaded = False
            if(not isLoaded):
                self.lock.acquire()
                try:
                    shm1 = SharedMemory(shm_string1)
                    shm2 = SharedMemory(shm_string2)
                    isLoaded = True
                except:
                    isLoaded = False
                if(not isLoaded):
                    self.cnt += 1
                    print(self.cnt)
                    local_level_step = 1
                    if(levelrange.shape[0]>1):
                        local_level_step = levelrange[1]-levelrange[0]
                    ExtractionLevelRange = slice(levelrange[0],levelrange[-1]+1,local_level_step)
                    #print("inserting:", mask_file_string1)
                    if(type == 0):
                        # load min and max and generate (min+max)/2 ("middle value") and 2/(max-min) ("inverse value range, scaled by 2")
                        m1 = np.reshape(np.fromfile(mask_file_string1), (17,721,1440))[ExtractionLevelRange]
                        m1 += np.reshape(np.fromfile(mask_file_string2), (17,721,1440))[ExtractionLevelRange]
                        m1 *= 0.5
                        m1 = m1.astype(np.float32)
                        m2 = np.reshape(np.fromfile(mask_file_string2), (17,721,1440))[ExtractionLevelRange]
                        m2 -= np.reshape(np.fromfile(mask_file_string1), (17,721,1440))[ExtractionLevelRange]
                        m2 = 2/m2
                        m2 = m2.astype(np.float32)
                    elif(type == 1):
                        # load mean
                        m1 = np.reshape(np.fromfile(mask_file_string1), (17,721,1440))[ExtractionLevelRange]
                        m1 = m1.astype(np.float32)
                        # load variance and save inverse root
                        m2 = np.reshape(np.fromfile(mask_file_string2), (17,721,1440))[ExtractionLevelRange]
                        m2 = 1/np.sqrt(m2)
                        m2 = m2.astype(np.float32)
                    shm1 = SharedMemory(name=shm_string1, create = True, size = m1.size*m1.itemsize)
                    shm2 = SharedMemory(name=shm_string2, create = True, size = m2.size*m2.itemsize)
                    # copy m1 and m2 data to the shared memory buffer
                    shm1buf = np.ndarray(m1.shape, dtype=m1.dtype, buffer = shm1.buf)
                    shm2buf = np.ndarray(m2.shape, dtype=m2.dtype, buffer = shm2.buf)
                    shm1buf[:] = m1[:]
                    shm2buf[:] = m2[:]
                    #print(shm1.size, m1.size*m1.itemsize, m1.itemsize, 721*1440*9*4)
                self.lock.release()
            # get local offsets
            local_lat_step = 1
            local_lon_step = 1
            if(latrange.shape[0]>1):
                local_lat_step = latrange[1]-latrange[0]
            if(lonrange.shape[0]>1):
                local_lon_step = lonrange[1]-lonrange[0]
            extraction_location = (slice(0, len(levelrange), 1),slice(latrange[0],latrange[-1]+1,local_lat_step),slice(lonrange[0],lonrange[-1]+1,local_lon_step))
            # read only access so we should not need to copy
            mask1 = np.ndarray((levelrange.shape[0],721,1440), dtype=np.float32, buffer = shm1.buf)[extraction_location]
            mask2 = np.ndarray((levelrange.shape[0],721,1440), dtype=np.float32, buffer = shm2.buf)[extraction_location]
            img -= mask1
            img *= mask2
        #return mask1, mask2

def getDateFromName(name):
    name = name.split("/")[-1]
    name = os.path.splitext(name)[0]
    return name[-11:-7], name[-7:-5], name[-5:-3], name[-2:]

def getLocalValues(rootgrp, latrange, lonrange, levelrange , lat_step, lon_step, warpmask):
        keys = np.asarray(rootgrp)
        xydim = ["longitude", "latitude"]
        if("longitude" in keys):
            pass
        elif("lon" in keys):
            xydim[0] = "lon"
        if("latitude" in keys):
            pass
        elif("lat" in keys):
            xydim[1] = "lat"

        # set Levelrange to all levels as default
        if(levelrange is None):
            levelrange = rootgrp["level"][:].astype(np.int32)
        if(lat_step is None):
            lat_step = (rootgrp[xydim[1]][1]-rootgrp[xydim[1]][0])
        if(lon_step is None):
            lon_step = (rootgrp[xydim[0]][1]-rootgrp[xydim[0]][0])
        # If warping is to be performed, we restrict ourselves to the patch that contains the warping area
        local_warpmask = None
        if(not warpmask is None):
            lat_reversed = lat_step < 0
            eps = 5.0
            minlat = max(int(np.min(warpmask[0]))-eps, -90.01)
            maxlat = min(int(np.max(warpmask[0]))+eps, 90)
            minlon = max(int(np.min(warpmask[1]))-eps, -180)
            maxlon = min(int(np.max(warpmask[1]))+eps, 180.01)
            latrange = (maxlat, minlat) if lat_reversed else (minlat, maxlat) 
            lonrange = (minlon, maxlon)
            local_warpmask = np.zeros_like(warpmask)
            local_warpmask[0] = (warpmask[0] - maxlat)/lat_step if lat_reversed else (warpmask[0] - minlat)/lat_step
            local_warpmask[1] = (warpmask[1] - minlon)/lon_step
            
        # set latrange to all lats as default
        if(latrange is None):
            val_latrange = rootgrp[xydim[1]][:]
        else:
            val_latrange = np.arange(latrange[0], latrange[1], lat_step)
        # set lonrange ato all lons as default
        if(lonrange is None):
            val_lonrange = rootgrp[xydim[0]][:]
        else:
            val_lonrange = np.arange(lonrange[0], lonrange[1], lon_step)

        local_latrange = np.nonzero(np.isin(rootgrp[xydim[1]],val_latrange, True))[0]
        local_lonrange = np.nonzero(np.isin(rootgrp[xydim[0]],val_lonrange, True))[0]
        local_levelrange = np.nonzero(np.isin(rootgrp["level"],levelrange, True))[0]

        return local_latrange, local_lonrange, local_levelrange, local_warpmask

def extractImageFromCDFh5pyChunkedSlim1dAfterNormGeneralDerivativeAndCache(rootgrp, variables , latrange , lonrange, levelrange, asHDF5):
    # levlrange is in L137 model levels, convert to the local matrix indices
    local_level_step = 1
    local_lat_step = 1
    local_lon_step = 1
    if(levelrange.shape[0]>1):
        local_level_step = levelrange[1]-levelrange[0]
    if(latrange.shape[0]>1):
        local_lat_step = latrange[1]-latrange[0]
    if(lonrange.shape[0]>1):
        local_lon_step = lonrange[1]-lonrange[0]
    
    levels = len(levelrange)
    singleLevelVariables= np.nonzero(np.isin(variables,['latitude','latitude_c','longitude','longitude_c','kmPerLon','kmPerLon_c','kmPerLat','kmPerLat_c'], True))[0].shape[0]
    mydat = np.zeros(((len(variables)-singleLevelVariables)*levels+singleLevelVariables,len(latrange),len(lonrange)))    

    if(asHDF5):
        chunk_shape = rootgrp['t'].chunks
        time_chunk = chunk_shape[0]
        level_chunk = chunk_shape[1]
        lat_chunk = chunk_shape[2]
        lon_chunk = chunk_shape[3]
    else:
        time_chunk = 1
        level_chunk = levels
        lat_chunk = latrange.shape[0]
        lon_chunk = lonrange.shape[0]

    origLevelRange = levelrange
    lr2 = len(levelrange)
    pos1 = np.where(levelrange%level_chunk == 0)[0]
    if(len(pos1) == 0):
        pos1 = np.concatenate(([0], pos1,[lr2]))
    if(pos1[0] != 0):
        pos1 = np.concatenate(([0],pos1))
    if(pos1[-1] != lr2):
        pos1 = np.concatenate((pos1,[lr2]))
    levels_to_extract_batches_from = [slice(levelrange[pos1[i]], levelrange[pos1[i+1]-1]+1, local_level_step) for i in range(len(pos1)-1)]
    levels_to_extract_batches_to = [slice(pos1[i], pos1[i+1], 1) for i in range(len(pos1)-1)]

    origLatRange = latrange
    lr2 = len(latrange)
    pos1 = np.where(latrange%lat_chunk == 0)[0]
    if(len(pos1)==0):
        pos1 = np.concatenate(([0],pos1,[lr2]))
    if(pos1[0] != 0):
        pos1 = np.concatenate(([0],pos1))
    if(pos1[-1] != lr2):
        pos1 = np.concatenate((pos1,[lr2]))
    latrange = [slice(latrange[pos1[i]], latrange[pos1[i+1]-1]+1, local_lat_step) for i in range(len(pos1)-1)]

    lr2 = len(lonrange)
    pos = np.where(lonrange%lon_chunk == 0)[0]
    if(len(pos)==0):
        pos = np.concatenate(([0],pos,[lr2]))
    if(pos[0] != 0):
        pos = np.concatenate(([0],pos))
    if(pos[-1] != lr2):
        pos = np.concatenate((pos,[lr2]))
    lonrange = [slice(lonrange[pos[i]], lonrange[pos[i+1]-1]+1, local_lon_step) for i in range(len(pos)-1)]

    # Generate Ranges to read from and write to
    levels_to_extract_from = levels_to_extract_batches_from
    levels_to_extract_to = levels_to_extract_batches_to

    source = np.array([[[(0,x ,lat,lon) for lat in latrange] for lon in lonrange] for x in levels_to_extract_from])
    target = np.array([[[(x,slice(pos1[j],pos1[j+1],1),slice(pos[i],pos[i+1],1)) for j in range(len(pos1)-1)] for i in range(len(pos)-1)] for x in levels_to_extract_to])

    # Read all requested variables
    idx = 0
    # index for the single level values
    svalIdx = 0
    cache = {}
    cached = {}
    for full_variable in variables:
        if(full_variable == "time" or full_variable == 'level'):
            continue
        
        # get Index Range 
        idxRange = slice(idx*levels,(idx+1)*levels,1)
        idx+=1
        if(full_variable in ['latitude','latitude_c','longitude','longitude_c','kmPerLon','kmPerLon_c','kmPerLat','kmPerLat_c']):
            idxRange = slice(mydat.shape[0]-svalIdx-1,mydat.shape[0]-svalIdx,1)
            svalIdx += 1
            # Special case, always at the last slot
            idx -= 1
        
        # read the variable
        readVariable(rootgrp, full_variable, mydat[idxRange], source, target, cache, cached, True, asHDF5)
        
    return mydat

def readVariable(rootgrp, full_variable, mydat, source, target, cache, cached, writeVar, asHDF5):    
    cacheThis = False

    # suffix _c indicates that this variable should be cached
    if(full_variable[-2:] == '_c'):
        cacheThis = True
        full_variable = full_variable[:-2]
    #If the variable is to be cached => generate the cache array
    if(cacheThis):
        if(full_variable in ['latitude','longitude','kmPerLon','kmPerLat']):
            cache[full_variable] = np.zeros_like(mydat[:,0])
        else:
            cache[full_variable] = np.zeros_like(mydat)
        cached[full_variable] = False

    # extract the extraction information from the variable string
    # find if a transformation function is used "(" ")"
    firstFunc = full_variable.find("(") 
    endFunc = full_variable.rfind(")")
    fname = full_variable[:firstFunc]
    if(fname == "pol"):
        # assume only a single "," exists
        targetVars = full_variable[firstFunc+1:endFunc].split(",")
        buff1 = readVariable(rootgrp, targetVars[0], mydat, source, target, cache, cached, False, asHDF5)
        buff2 = readVariable(rootgrp, targetVars[1], mydat, source, target, cache, cached, False, asHDF5)
        return getPolar(buff1, buff2, mydat, True, writeVar)
    elif(fname == "abs"):
        targetVars = full_variable[firstFunc+1:endFunc].split(",")
        buff1 = readVariable(rootgrp, targetVars[0], mydat, source, target, cache, cached, False, asHDF5)
        buff2 = readVariable(rootgrp, targetVars[1], mydat, source, target, cache, cached, False, asHDF5)
        return getPolar(buff1, buff2, mydat, False, writeVar)
    elif(fname == "delta_u"):
        targetVar = full_variable[firstFunc+1:endFunc]
        buff1 = readVariable(rootgrp, targetVar, mydat, source, target, cache, cached, False, asHDF5)
        return getDerivative(buff1, mydat, 2, writeVar, None)
    elif(fname == "delta_l"):
        targetVar = full_variable[firstFunc+1:endFunc]
        buff1 = readVariable(rootgrp, targetVar, mydat, source, target, cache, cached, False, asHDF5)
        return getDerivative(buff1, mydat, -2, writeVar, cache['latitude'])
    elif(fname == "delta_v"):
        targetVar = full_variable[firstFunc+1:endFunc]
        buff1 = readVariable(rootgrp, targetVar, mydat, source, target, cache, cached, False, asHDF5)
        return getDerivative(buff1, mydat, 1, writeVar, None)
    elif(fname == "base"):
        targetVar = full_variable[firstFunc+1:endFunc]
        return readVariable(rootgrp, targetVar, mydat, source, target, cache, cached, writeVar, asHDF5)
    else:
        if(asHDF5):
            return getVariable(rootgrp, full_variable, mydat, source, target,cache, cached, cacheThis, writeVar)
        else:
            return getVariableCDF(rootgrp, full_variable, mydat, source, target,cache, cached, cacheThis, writeVar)
    
def getVariable(rootgrp, variable, mydat, source, target, cache, cached, cacheThis, writeVar):
    if(variable in cached and cached[variable]):
        if(writeVar):
            mydat = cache[variable]
        return cache[variable]
    else:
        # variable shall be written, then the buffer is the target destination
        if(writeVar):
            tgtBuffer = mydat
        # variable shall not be written, then the buffer is a copy of the target destination
        else:
            tgtBuffer = np.zeros_like(mydat)
    

        if(variable == 'ept' or variable == 'dewt'):
            add_off = (rootgrp['t'].attrs['add_offset'],rootgrp['q'].attrs['add_offset'], rootgrp['sp'].attrs['add_offset'])
            scal_fac = (rootgrp['t'].attrs['scale_factor'],rootgrp['q'].attrs['scale_factor'], rootgrp['sp'].attrs['scale_factor'])
        elif(variable[-4:] == '2pol' or variable[-4:] == '2abs'):
            add_off = (rootgrp[partialvars[0]].attrs['add_offset'],rootgrp[partialvars[1]].attrs['add_offset'])
            scal_fac = (rootgrp[partialvars[0]].attrs['scale_factor'],rootgrp[partialvars[1]].attrs['scale_factor'])
        elif(variable == 'latitude' or variable == 'lat' or variable == 'longitude' or variable == 'lon' or variable == 'kmPerLon' or variable == 'kmPerLat'):
            pass
        else:
            if("add_offset" in rootgrp[variable].attrs):
                add_off = rootgrp[variable].attrs['add_offset']
            if("scale_factor" in rootgrp[variable].attrs):
                scal_fac = rootgrp[variable].attrs['scale_factor']

    
        # Read derived variables
        # Variables depending on temperature (t), surface_pressure (sp) and specific humidity (q)
        # ept: equivalent potential temperature
        # dewt: dewpoint temperature
        if(variable == 'ept' or variable == 'dewt'):
            paBuffer = 0
            tBuffer = 0
            qBuffer = 0
            if('sp' in cached and cached['sp']):
                paBuffer = cache['sp']
            else:
                paBuffer = np.zeros_like(mydat)
                for lon_batch in range(source.shape[1]):
                    for lat_batch in range(source.shape[2]):
                        lsource = (0, *source[0,lon_batch,lat_batch][-2:])
                        ltarget = (-1, *target[0,lon_batch,lat_batch][-2:])
                        readHDF5VariablePlain(rootgrp['sp'], paBuffer, lsource, ltarget)
                scaleArray(paBuffer, scal_fac[2], add_off[2])
                levelrange = rootgrp["level"][slice(source[0,0,0][1].start, source[-1,0,0][1].stop,source[0,0,0][1].step)].astype(np.int32)
                calculatePressureFromML(paBuffer, levelrange)
            if('t' in cached and 'q' in cached and cached['t'] and cached['q']):
                tBuffer = cache['t']
                qBuffer = cache['q']
            else:
                tBuffer = np.zeros_like(mydat)
                qBuffer = np.zeros_like(mydat)
                for level_batch in range(source.shape[0]):
                    for lon_batch in range(source.shape[1]):
                        for lat_batch in range(source.shape[2]):
                            readHDF5VariablePlain(rootgrp['t'], tBuffer, source[level_batch][lon_batch][lat_batch], target[level_batch][lon_batch][lat_batch])
                            readHDF5VariablePlain(rootgrp['q'], qBuffer, source[level_batch][lon_batch][lat_batch], target[level_batch][lon_batch][lat_batch])
                scaleArray(tBuffer, scal_fac[0], add_off[0])
                scaleArray(qBuffer, scal_fac[1], add_off[1])
            if(variable == 'ept'):
                mydat[:] = equivalentPotentialTemp(tBuffer, qBuffer, paBuffer)
                #mydat[:] = ept(tBuffer, qBuffer, paBuffer)
            elif(variable == 'dewt'):
                mydat[:] = dewpointTemp(tBuffer, qBuffer, paBuffer)
        # Read special variable latitude
        elif(variable == 'latitude' or variable == 'kmPerLon'):
            origLatRange = np.arange(source[0,0,0][2].start, source[0,0,-1][2].stop)
            tmpBuffer = rootgrp['latitude'][origLatRange]
            if(variable == 'kmPerLon'):
                tmpBuffer = np.abs(tmpBuffer)
                tmpBuffer = np.reshape(LatTokmPerLon(tmpBuffer), (tmpBuffer.shape[0], 1))
                tmpBuffer = np.clip(tmpBuffer, 0.1, 30)/27.7762
            else:
                tmpBuffer = np.reshape(tmpBuffer, (tmpBuffer.shape[0], 1))
            tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
        elif(variable == 'longitude'):
            origLonRange = np.arange(source[0,0,0][3].start, source[0,-1,0][3].stop)
            tmpBuffer = rootgrp['longitude'][origLonRange]
            tmpBuffer = np.reshape(tmpBuffer, (1, tmpBuffer.shape[0]))
            tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
        # there is no distortion in latitudinal direction
        elif(variable == 'kmPerLat'):
            tgtBuffer = np.ones(tgtBuffer.shape)
        elif(variable == 'sp'):
            for lon_batch in range(source.shape[1]):
                for lat_batch in range(source.shape[2]):
                    lsource = (0, *source[0,lon_batch,lat_batch][-2:])
                    ltarget = (-1, *target[0,lon_batch,lat_batch][-2:])
                    readHDF5VariablePlain(rootgrp[variable], tgtBuffer, lsource, ltarget)
                    # scale the read values to their true range
            scaleArray(tgtBuffer, scal_fac, add_off)
            levelrange = rootgrp["level"][slice(source[0,0,0][1].start, source[-1,0,0][1].stop,source[0,0,0][1].step)].astype(np.int32)
            calculatePressureFromML(tgtBuffer, levelrange)

        # Read in a standard variable
        else:
            for level_batch in range(source.shape[0]):
                for lon_batch in range(source.shape[1]):
                    for lat_batch in range(source.shape[2]):
                        readHDF5VariablePlain(rootgrp[variable], tgtBuffer, source[level_batch][lon_batch][lat_batch], target[level_batch][lon_batch][lat_batch])
            
            # scale the read values to their true range
            scaleArray(tgtBuffer, scal_fac, add_off)

        if(cacheThis):
            cache[variable] = tgtBuffer
            cached[variable] = True
        return tgtBuffer

def getVariableCDF(rootgrp, variable, mydat, source, target, cache, cached, cacheThis, writeVar):
        if(variable in cached and cached[variable]):
            if(writeVar):
                mydat = cache[variable]
            return cache[variable]
        else:
            # variable shall be written, then the buffer is the target destination
            if(writeVar):
                tgtBuffer = mydat
            # variable shall not be written, then the buffer is a copy of the target destination
            else:
                tgtBuffer = np.zeros_like(mydat)

        
            # Read derived variables
            # Variables depending on temperature (t), surface_pressure (sp) and specific humidity (q)
            # ept: equivalent potential temperature
            # dewt: dewpoint temperature
            if(variable == 'ept' or variable == 'dewt'):
                paBuffer = 0
                tBuffer = 0
                qBuffer = 0
                if('sp' in cached and cached['sp']):
                    paBuffer = cache['sp']
                else:
                    paBuffer = np.zeros_like(mydat)
                    for lon_batch in range(source.shape[1]):
                        for lat_batch in range(source.shape[2]):
                            lsource = (0, *source[0,lon_batch,lat_batch][-2:])
                            ltarget = (-1, *target[0,lon_batch,lat_batch][-2:])
                            readCDFVariable(rootgrp['sp'], paBuffer, lsource, ltarget)
                    levelrange = rootgrp["level"][slice(source[0,0,0][1].start, source[-1,0,0][1].stop,source[0,0,0][1].step)].astype(np.int32)
                    calculatePressureFromML(paBuffer, levelrange)
                if('t' in cached and 'q' in cached and cached['t'] and cached['q']):
                    tBuffer = cache['t']
                    qBuffer = cache['q']
                else:
                    tBuffer = np.zeros_like(mydat)
                    qBuffer = np.zeros_like(mydat)
                    for level_batch in range(source.shape[0]):
                        for lon_batch in range(source.shape[1]):
                            for lat_batch in range(source.shape[2]):
                                readCDFVariable(rootgrp['t'], tBuffer, source[level_batch][lon_batch][lat_batch], target[level_batch][lon_batch][lat_batch])
                                readCDFVariable(rootgrp['q'], qBuffer, source[level_batch][lon_batch][lat_batch], target[level_batch][lon_batch][lat_batch])
                if(variable == 'ept'):
                    mydat[:] = equivalentPotentialTemp(tBuffer, qBuffer, paBuffer)
                elif(variable == 'dewt'):
                    mydat[:] = dewpointTemp(tBuffer, qBuffer, paBuffer)
            # Read special variable latitude
            elif(variable == 'latitude' or variable == 'kmPerLon'):
                origLatRange = np.arange(source[0,0,0][2].start, source[0,0,-1][2].stop)
                tmpBuffer = rootgrp['latitude'][origLatRange]
                if(variable == 'kmPerLon'):
                    tmpBuffer = np.abs(tmpBuffer)
                    tmpBuffer = np.reshape(LatTokmPerLon(tmpBuffer), (tmpBuffer.shape[0], 1))
                    tmpBuffer = np.clip(tmpBuffer, 0.1, 30)/27.7762
                else:
                    tmpBuffer = np.reshape(tmpBuffer, (tmpBuffer.shape[0], 1))
                tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
            elif(variable == 'longitude'):
                origLonRange = np.arange(source[0,0,0][3].start, source[0,-1,0][3].stop)
                tmpBuffer = rootgrp['longitude'][origLonRange]
                tmpBuffer = np.reshape(tmpBuffer, (1, tmpBuffer.shape[0]))
                tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
            elif(variable == 'sp'):
                for lon_batch in range(source.shape[1]):
                    for lat_batch in range(source.shape[2]):
                        lsource = (0, *source[0,lon_batch,lat_batch][-2:])
                        ltarget = (-1, *target[0,lon_batch,lat_batch][-2:])
                        readCDFVariable(rootgrp[variable], tgtBuffer, lsource, ltarget)
                        # scale the read values to their true range
                levelrange = rootgrp["level"][slice(source[0,0,0][1].start, source[-1,0,0][1].stop,source[0,0,0][1].step)].astype(np.int32)
                calculatePressureFromML(tgtBuffer, levelrange)

            # Read in a standard variable
            else:
                for level_batch in range(source.shape[0]):
                    for lon_batch in range(source.shape[1]):
                        for lat_batch in range(source.shape[2]):
                            readCDFVariable(rootgrp[variable], tgtBuffer, source[level_batch][lon_batch][lat_batch], target[level_batch][lon_batch][lat_batch])

            if(cacheThis):
                cache[variable] = tgtBuffer
                cached[variable] = True
            return tgtBuffer


def warpImage(img, mask):
    out = np.zeros((img.shape[0], *mask.shape[1:]))
    for level in range(img.shape[0]):
        out[level] = map_coordinates(img[level], mask, mode='wrap')
    return out

def getPolar(uBuffer, vBuffer, mydat, angle, writeVar):
    if(writeVar):
        tgtBuffer = mydat
    else:
        tgtBuffer = np.zeros_like(mydat)
    # u and v buffer are filled
    if(angle):
        # The angle deviating counter clockwise from the east-west axis
        tgtBuffer[:] = np.abs(np.angle(uBuffer+1j*vBuffer))
    else:
        # The absolute signed by the north south direction
        tgtBuffer[:] = np.abs(uBuffer+1j*vBuffer)*np.sign(vBuffer)
    return tgtBuffer

def getDerivative(buffer, mydat, axis, writeVar, scaling ):
    # Post processing (e.g. derivative )
    if(writeVar):
        tgtBuffer = mydat
    else:
        tgtBuffer = np.zeros_like(mydat)
    if(axis == -2):
        tgtBuffer[:] = np.gradient(buffer, axis = 2)/scaling
    else:
        tgtBuffer[:] = np.gradient(buffer, axis= axis)
    return tgtBuffer

def readCDFVariable(in_Buffer, out_buffer, in_slice, out_slice):
    out_buffer[tuple(out_slice)] = in_Buffer[tuple(in_slice)]

def readHDF5Variable(dataset_in,array_out, in_slice, out_slice, scal_fac, add_off):
    dataset_in.read_direct(array_out, in_slice, out_slice)
    array_out *= scal_fac
    array_out += add_off

def readHDF5VariablePlain(dataset_in,array_out, in_slice, out_slice):
    #array_out[out_slice] = dataset_in[in_slice]
    dataset_in.read_direct(array_out, tuple(in_slice), tuple(out_slice))

def LatTokmPerLon(lat_data, lonResolution = 0.25):
    unit = 1000
    radius = 6365831.0
    r = (radius*np.cos(lat_data/180*np.pi))/unit
    lat_data = r*2*np.pi*lonResolution / 360
    return lat_data

def calculatePressureFromML(data, levels):
    lc = L137Calculator()
    lc.getPressureAtMultipleLevels(data[-1], levels, data)

def scaleArray(data, scal_fac, add_off):
    data *= scal_fac
    data += add_off

def dewpointTemp(temp, humidity, level_range = slice(0,6,1)):
    K3 = 243.12
    K2 = 17.62
    K1 = 6.112
    cels = temp-273.15
    psaett = K1 * np.exp((K2*cels)/(K3+cels))
    PaPerLevel = [722.9795, 795.6396, 856.8376, 986.6036, 1002.2250, 1013.2500]
    PaPerLevel = PaPerLevel[level_range]
    #PaPerLevel = [986.6036, 1002.2250, 1013.2500]
    #PaPerLevel = [722.9795, 795.6396, 856.8376]
    #ph = np.zeros(humidity.shape)
    S = np.zeros_like(humidity)
    for i in range(len(PaPerLevel)):
        #ph[i] = humidity[i]*PaPerLevel[i]
        S[i] = 0.622*psaett[i] / ((PaPerLevel[i]-0.378*psaett[i]))
    relHum = humidity/S
    lnh = np.log(relHum, where=(relHum>0))

    #dewP1 = K3* np.log((ph)/((0.622+humidity)*K1))
    #dewP2 = K2 - np.log((ph)/((0.622+humidity)*K1))
    #dewP = dewP1/dewP2
    #divisor = K3+cels
    
    dewP1 = K2*cels / (K3+cels) + lnh
    dewP2 = K2*K3 / (K3+cels) - lnh

    dewP = K3* dewP1/dewP2
    return dewP

def equivalentPotentialTempNew(temp, humidity, PaPerLevel):
    
    cp = 1004.82
    Rd = 287.05
    equiPotTemp = temp[:]
    L = 2257000
    m = humidity
    PaPerLevel = [722.9795, 795.6396, 856.8376, 986.6036, 1002.2250, 1013.2500]
    #PaPerLevel = PaPerLevel[level_range]
    for i in range(len(PaPerLevel)):
        equiPotTemp[i] *= pow((1000/PaPerLevel[i]), Rd/cp)*np.exp(L*m[i] / (cp*temp[i]))
    return equiPotTemp
 

def equivalentPotentialTemp(t, q, PaPerLevel):
    #pu = units.Quantity(p, "Pa")
    #tu = units.Quantity(t, "K")
    #dewp = dewpoint_from_specific_humidity(pu, tu, q)
    #return equivalent_potential_temperature(pu, tu, dewp)
    '''
    # celsius temp
    ctemp = t-273.15

    # constants
    cp = 1.00482
    cw = 4.18674
    Rd = 287.05

    # L value
    L = 2500.78-2.325734*ctemp

    #equivalent temp
    ctemp += q*(L/(cp+q*cw)) + 273.15

    # precalculate exponent
    cp *= 1000
    val = Rd/cp

    #equivalent potential temp (this is very slow! approx 3.5 seconds)
    # multiplication by 100 to get Pa out of hPa
    ctemp *= np.power(100000/PaPerLevel, val)

    return ctemp
    '''

def getValueRanges(variable):
    minval,maxval = 0,1
    # pseudo-normalize variables 
    if(variable == 't'):
        minval, maxval = (273.15-60), 273.15+45
    elif(variable == 'ept'):
        minval, maxval = 273.15-60, 273.15+90
    elif(variable == 'dewt'):
        minval, maxval = -90, 40
    elif(variable == 'dtu' or variable == 'delta_t_udir' or variable == "delta_u(t)" or variable == "delta_l(t)"):
        minval, maxval = -14, 14
    elif(variable == 'dtv' or variable == 'delta_t_vdir' or variable == "delta_v(t)"):
        minval, maxval = -14, 14
    elif(variable == 'q'):
        minval, maxval = 0, 0.025
    elif(variable == 'delta_q_udir' or variable == 'delta_q_vdir' or variable == 'delta_q_ldir' or variable == "delta_u(q)" or variable == "delta_v(q)"):
        minval, maxval = -0.01, 0.01
    elif(variable == 'u'):
        minval, maxval = -60, 60
    elif(variable == 'v'):
        minval, maxval = -60, 60
    elif(variable[:3] == "pol"):
        minval, maxval = 0, np.pi
    elif(variable == "abs(u,v)"):
        minval, maxval = -60, 60
    elif(variable == "abs(delta_u(t),delta_v(t))"):
        minval, maxval = -14, 14
    elif(variable == "abs(delta_u(q),delta_v(q))"):
        minval, maxval = -0.01, 0.01
    elif(variable == 'delta_u_udir' or variable == 'delta_u_vdir' or variable == 'delta_u_ldir'):
        minval, maxval = -22, 22
    elif(variable == 'delta_v_udir' or variable == 'delta_v_vdir' or variable == 'delta_v_ldir'):
        minval, maxval = -22, 22
    elif(variable == 'delta_u(u)' or variable == 'delta_v(u)' or variable == 'delta_l(u)'):
        minval, maxval = -22, 22
    elif(variable == 'delta_u(v)' or variable == 'delta_v(v)' or variable == 'delta_l(v)'):
        minval, maxval = -22, 22
    elif(variable == 'w'):
        minval, maxval = -15, 15
    elif(variable == 'sp'):
        minval, maxval = 400*100, 1050*100
    elif(variable == 'kmPerLon'):
        minval, maxval = -1, 1
    elif(variable == 'kmPerLat'):
        minval, maxval = -1, 1
    elif(variable == 'latitude'):
        minval, maxval = -90, 90
    elif(variable == 'longitude'):
        minval, maxval = -180, 180
    return minval,maxval


def getMeanVar(variable):
    mean,var = 0.5,1
    # pseudo-normalize variables 
    if(variable == 't'):
        mean, var = 2.75355461e+02, 3.20404803e+02
    elif(variable == 'q'):
        mean, var = 5.57926815e-03, 2.72627785e-05 
    elif(variable == 'u'):
        mean, var = 1.27024432, 6.74232481e+01
    elif(variable == 'v'):
        mean, var = 1.0213897e-01, 4.36244384e+01
    elif(variable == 'w'):
        mean, var = 5.87718196e-03, 4.77972548e-02
    elif(variable == 'sp'):
        mean, var = 8.65211548e+04, 1.49460630e+08
    elif(variable == 'kmPerLon'):
        mean, var = 0.64, 0.09
    return mean, var

# Very simple CDF extractor
def extractImageFromCDFtmp(filename, vars, latrange, lonrange, levelrange = None, TotalLatRange = 721, TotalLonRange = 1440, resolution = 0.25):
    rootgrp = Dataset(os.path.realpath(filename), "r", format="NETCDF4", parallel=False)   

     # set Levelrange to all levels as default
    if(levelrange is None):
        levelrange = rootgrp["level"][:].astype(np.int32)
    # levelrange is in L137 model levels, convert to the local matrix indices
    local_levelrange = np.nonzero(np.isin(rootgrp["level"], levelrange, True))[0]
    local_levelrange = np.arange(local_levelrange[0], local_levelrange[-1]+1,local_levelrange[1]-local_levelrange[0])
    levels = len(levelrange)
    mydat = np.zeros((len(vars),levels,len(latrange),len(lonrange)))
    


    # Adjust lonrange array position to the offset of the data
    offset = 0
    if("longitude" in rootgrp.variables):
        offset = rootgrp["longitude"][0]
    elif("lon" in rootgrp.variables):
        offset = rootgrp["lon"][0]
    lonrange = (lonrange-int(offset/resolution))%TotalLonRange

    # find breakpoint of data lonrange (e.g. 359 degree -> 0 degree)
    # This is purely done for better data slicing
    lr2 = len(lonrange)
    pos = np.where(lonrange == 0)[0]
    if(len(pos>0)):
        lr2 = pos[0]

    # Prepare temporary vectors to save intermediate values needed for derived variables
    tmp1 = np.zeros_like(mydat[0])
    tmp2 = np.zeros_like(mydat[0])
    tmpPa = np.zeros_like(mydat[0])
    tmpfilled = False

    # Read all requested variables
    for idx, variable in enumerate(vars):
        
        # Normalization offsets
        minval,maxval = getValueRanges(variable)
        valRange = (maxval-minval)

        # Read variables
        if(variable == 'ept' or variable == 'dewt'):
            if(not tmpfilled):
                if(offset==0):
                    tmp1[:,:,:lr2] = rootgrp['t'][0, local_levelrange, latrange, lonrange[:lr2]]
                    tmp1[:,:,lr2:] = rootgrp['t'][0, local_levelrange, latrange, lonrange[lr2:]] 
                    tmp2[:,:,:lr2] = rootgrp['q'][0, local_levelrange, latrange, lonrange[:lr2]]
                    tmp2[:,:,lr2:] = rootgrp['q'][0, local_levelrange, latrange, lonrange[lr2:]]
                else:
                    tmp1 = rootgrp['t'][0, local_levelrange, latrange, lonrange]
                    tmp2 = rootgrp['q'][0, local_levelrange, latrange, lonrange]
                tmpfilled = True
            if(variable == 'ept'):
                mydat[idx] = equivalentPotentialTemp(tmp1, tmp2, tmpPa)
            elif(variable == 'dewt'):
                mydat[idx] = dewpointTemp(tmp1,tmp2, tmpPa)
        elif(variable == 'dtu' or variable == 'dtv'):
            axis = 1
            if(variable == 'dtv'):
                axis = 2
            if(not tmpfilled):
                if(offset==0):
                    mydat[idx,:,:,:lr2] = rootgrp['t'][0, local_levelrange, latrange, lonrange[:lr2]]
                    mydat[idx,:,:,lr2:] = rootgrp['t'][0, local_levelrange, latrange, lonrange[lr2:]]
                else:
                    mydat[idx] = rootgrp['t'][0, local_levelrange, latrange, lonrange]
                mydat[idx] = np.gradient(mydat[idx], axis= axis) 
            else:
                mydat[idx] = np.gradient(tmp1, axis= axis) 
        elif(variable == 'sp'):
            if(offset == 0):
                mydat[idx,0,:,:lr2] = rootgrp['sp'][0,latrange,lonrange[:lr2]]
                mydat[idx,0,:,lr2:] = rootgrp['sp'][0,latrange,lonrange[lr2:]]
            else:
                mydat[idx,-1] = rootgrp['sp'][0,latrange,lonrange]
                calculatePressureFromML(mydat[idx], levelrange)
                tmpPa[:] = mydat[idx]
        else:
            # temperature and precipation at levels between ~700 and ~850 hPa (3087m to 1328m) and sea level
            # other variables at 10m (1012 hPa) (~ sea level)
            if((variable == 'q') and tmpfilled):
                mydat[idx] = tmp2[:]
            elif(( variable == 't') and tmpfilled):
                mydat[idx] = tmp1[:]
            else:
                if(offset==0):
                    mydat[idx,:,:,:lr2] = rootgrp[variable][0,local_levelrange,latrange,lonrange[:lr2]]
                    mydat[idx,:,:,lr2:] = rootgrp[variable][0,local_levelrange,latrange,lonrange[lr2:]]
                else:
                    mydat[idx,:,:,:] = rootgrp[variable][0,local_levelrange,latrange,lonrange]
        # Normalize variable 
        mydat[idx] -= minval+valRange/2
        mydat[idx] *= (2/valRange)
    rootgrp.close()       
    return mydat

