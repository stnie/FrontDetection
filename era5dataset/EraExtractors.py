import numpy as np
import os
import random

from .ERA5Reader.readNetCDF import CDFReader, ETHReader, LatTokmPerLon

class DefaultEraExtractor():
    # Determine which variables should be extracted
    def __init__(self, variables = ['t','q','u','v','w'],normType = 0):
        self.variables = variables
        # Create a CDF Reader using h5py and min max normalization
        self.reader = CDFReader(0, normType = normType)
    def __call__(self, filename, latrange, lonrange, levelrange, seed = 0, warpmask = None):
        if isinstance(filename, list):
            print("Currently not correct!")
            exit(1)
            return self.reader.read(filename, self.variables, latrange,lonrange, levelrange)
        return self.reader.read(filename, self.variables, latrange,lonrange, levelrange, warpmask = warpmask)


class ETHEraExtractor():
    # Determine which variables should be extracted
    def __init__(self, variables = ['FRONT'], normType = 0):
        self.variables = variables
        self.reader = ETHReader()
    def __call__(self, filename, latrange, lonrange, levelrange, seed = 0, warpmask = None):
        if isinstance(filename, list):
            print("Currently not correct!")
            exit(1)
            return self.reader.read(filename, latrange, lonrange)
        return self.reader.read(filename, latrange, lonrange)
        


# EraExtractor that inverses gradients if a corresponding flip occurs during transformation
class DerivativeFlippingAwareEraExtractor():
    # Determine which variables should be extracted
    def __init__(self, variables = ['t','q','u','v','w'], horizontal_indices = None, vertical_indices = None, fliprate = 0.5, horizontal_flipPos = 0, vertical_flipPos = 1, normType = 0, sharedObj = None):
        # Create a CDF Reader using h5py and min max normalization
        self.reader = CDFReader(0, normType = normType, sharedObj = sharedObj)
        
        self.variables = variables
        latoffs = []
        off = 0
        
        self.v_dir_derivatives = horizontal_indices
        self.u_dir_derivatives = vertical_indices

        self.horizontal_flipPos = horizontal_flipPos
        self.vertical_flipPos = vertical_flipPos
        self.fliprate = fliprate
        if(not self.u_dir_derivatives is None):
            for idx in self.u_dir_derivatives:
                print(self.variables[idx])
            print("and:")
        if(not self.v_dir_derivatives is None):
            for idx in self.v_dir_derivatives:
                print(self.variables[idx])
    def __call__(self, filename, latrange, lonrange, levelrange, flipseed, warpmask = None):
        if isinstance(filename, list):
            print("Currently not correct!")
            exit(1)
            return self.reader.read(filename, self.variables, latrange,lonrange, levelrange)
        tmp = self.reader.read(filename, self.variables, latrange,lonrange, levelrange, warpmask = warpmask)
        # simulate flipping to determine whether or not the gradients need be inversed
        if(self.fliprate > 0):
            random.seed(flipseed)
            for idx in range (max(self.horizontal_flipPos,self.vertical_flipPos)+1):
                if(idx == self.horizontal_flipPos):
                    number = random.random()
                    horizontal_flip = number < self.fliprate
                if(idx == self.vertical_flipPos):
                    number = random.random()
                    vertical_flip = number < self.fliprate
            num_levels = levelrange.shape[0]

            if(horizontal_flip):
                for idx in self.v_dir_derivatives:
                    tmp[idx*num_levels:(idx+1)*num_levels] *= -1
            if(vertical_flip):
                for idx in self.u_dir_derivatives:
                    tmp[idx*num_levels:(idx+1)*num_levels] *= -1
        return tmp
    def __repr__(self):
        myString = "DerivativeFlippingAwareEraExtractor\n"
        myString += str(self.__dict__)
        return myString


# EraExtractor that inverses gradients if a corresponding flip occurs during transformation
class DerivativeFlippingAwareMultiTimeEraExtractor():
    # Determine which variables should be extracted
    def __init__(self, variables = ['t','q','u','v','w'], horizontal_indices = None, vertical_indices = None, fliprate = 0.5, horizontal_flipPos = 0, vertical_flipPos = 1, normType = 0, sharedObj = None):
        # Create a CDF Reader using h5py and min max normalization
        self.reader = CDFReader(0, normType = normType, sharedObj = sharedObj)
        
        self.variables = variables
        latoffs = []
        off = 0
        
        self.v_dir_derivatives = horizontal_indices
        self.u_dir_derivatives = vertical_indices

        self.horizontal_flipPos = horizontal_flipPos
        self.vertical_flipPos = vertical_flipPos
        self.fliprate = fliprate
        if(not self.u_dir_derivatives is None):
            for idx in self.u_dir_derivatives:
                print(self.variables[idx])
            print("and:")
        if(not self.v_dir_derivatives is None):
            for idx in self.v_dir_derivatives:
                print(self.variables[idx])
    def __call__(self, filename, latrange, lonrange, levelrange, flipseed, warpmask = None):
        if isinstance(filename, list):
            print("Currently not correct!")
            exit(1)
            return self.reader.read(filename, self.variables, latrange,lonrange, levelrange)
        
        files = [self.getNextSnapshot(filename, x) for x in range(-6,1,6)]
        tmp = self.reader.read(files[0], self.variables, latrange,lonrange, levelrange, warpmask = warpmask)
        timeStampLevels = tmp.shape[0]
        # read the remainig timestamps
        for fname in files[1:]:
            tmp = np.concatenate((tmp, self.reader.read(fname, self.variables, latrange,lonrange, levelrange, warpmask = warpmask)), axis=0)
        
        # simulate flipping to determine whether or not the gradients need be inversed
        if(self.fliprate > 0):
            random.seed(flipseed)
            for idx in range (max(self.horizontal_flipPos,self.vertical_flipPos)+1):
                if(idx == self.horizontal_flipPos):
                    number = random.random()
                    horizontal_flip = number < self.fliprate
                if(idx == self.vertical_flipPos):
                    number = random.random()
                    vertical_flip = number < self.fliprate
    
            num_levels = levelrange.shape[0]
            # flip for each timestamp
            for timestamp in range(len(files)):
                tsoff = timestamp*timeStampLevels
                if(horizontal_flip):
                    for idx in self.v_dir_derivatives:
                        tmp[tsoff+idx*num_levels:tsoff+(idx+1)*num_levels] *= -1
                if(vertical_flip):
                    for idx in self.u_dir_derivatives:
                        tmp[tsoff+idx*num_levels:tsoff+(idx+1)*num_levels] *= -1
        return tmp
        
    def getNextSnapshot(self,filename, offset):
        name, ext = os.path.splitext(filename) 
        baseTime = int(name[-2:])
        baseDay = int(name[-5:-3])
        baseMonth = int(name[-7:-5])
        baseYear = int(name[-11:-7])
        dayOff = 0
        monthOff = 0
        yearOff = 0
        newTime = (baseTime+offset)
        if(newTime >=24):
            dayOff = 1
        if(newTime < 0):
            dayOff = -1
        newDay = baseDay+dayOff
        #Our Month has this many days
        days = 30
        if baseMonth == 2:
            days = 28
        elif baseMonth in [1,3,5,7,8,10,12]:
            days = 31
        if(newDay > days):
            monthOff = 1
        if(newDay < 1):
            monthOff -= 1

        newMonth = baseMonth+monthOff

        if(newMonth > 12):
            yearOff = 1
        if(newMonth < 1):
            yearOff = -1
        newYear = baseYear+yearOff
        newMonth = newMonth%12
        
        days = 30
        if newMonth == 2:
            days = 28
        elif newMonth in [1,3,5,7,8,10,12]:
            days = 31
        
        newDay = newDay%days
        newTime = newTime % 24
        
        fn = name[:-7]+"{:02d}{:02d}_{:02d}".format(newMonth, newDay, newTime)+ext
        if(not os.path.isfile(fn)):
            fn = name[:-7]+"{:02d}{:02d}_{:02d}".format(baseMonth, baseDay, baseTime)+ext
        return fn
    def __repr__(self):
        myString = "DerivativeFlippingAwareEraExtractor\n"
        myString += str(self.__dict__)
        return myString

class DoubleEraExtractor():
    # Determine which variables should be extracted
    def __init__(self, variables = ['t','q','u','v','w']):
        self.variables = variables
    def __call__(self, filename, latrange, lonrange, levelrange):
        if isinstance(filename, list):
            print("Currently not correct!")
            exit(1)
        fn2 = self.getNextSnapshot(filename, 6)
        # Next File exists, return concatenation of both files
        if(os.path.exists(fn2)):
            return np.concatenate((extractFromRange1dAfterNormGenDerAndCache(filename, self.variables, latrange,lonrange, levelrange),extractFromRange1dAfterNorm(fn2, self.variables, latrange,lonrange, levelrange)), axis=0)
        # Next File does not exists, return same file twice
        else: 
            return np.concatenate((extractFromRange1dAfterNormGenDerAndCache(filename, self.variables, latrange,lonrange, levelrange),extractFromRange1dAfterNorm(filename, self.variables, latrange,lonrange, levelrange)), axis = 0)
    
    def getNextSnapshot(filename, offset):
        name, ext = os.path.splitext(filename) 
        oldTime = int(name[-2:])
        oldDay = int(name[-5:-3])
        oldMonth = int(name[-7:-5])
        newTime = (oldTime+6)%24
        newDay = int(name[-5:-3])
        newMonth = int(name[-7:-5])
        # we are at a new day
        days = 30
        if newMonth == 2:
            days = 28
        elif newMonth in [1,3,5,7,8,10,12]:
            days = 31
        if newTime < oldTime:
            newDay = (oldDay%days)+1
            # new month begins
            if(newDay < oldDay):
                newMonth = oldMonth+1

        fn = name[:-7]+"{:02d}{:02d}_{:02d}".format(newMonth, newDay, newTime)+ext
        return fn
