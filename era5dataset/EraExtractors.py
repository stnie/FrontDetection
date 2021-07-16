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
    def __init__(self, variables = ['t','q','u','v','w'], horizontal_indices = None, vertical_indices = None, fliprate = 0.5, horizontal_flipPos = 0, vertical_flipPos = 1, normType = 0, sharedObj = None, ftype = 0):
        # Create a CDF Reader using h5py and min max normalization
        self.reader = CDFReader(ftype, normType = normType, sharedObj = sharedObj)
        
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


