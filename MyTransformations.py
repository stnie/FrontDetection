import random
import numpy as np



class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img):

        # THIS FLIPS 0 1 2 to  3 4 5
        #            3 4 5     0 1 2
        prob = random.random()
        if prob < self.p:
            return np.flip(img,0)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img):

        prob = random.random()
        if prob < self.p:
            return np.flip(img,1)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHorizontalCoordsFlip(object):

    def __init__(self, size, p=0.5):
        super().__init__()
        self.p = p
        self.size = size

    def __call__(self, coords):

        # THIS FLIPS 0 1 2 to  3 4 5
        #            3 4 5     0 1 2
        # Assumption Sizes and coords are always given in
        # lat lon 
        # above examples has len(lat) = self.size[0] = 2, len(lon) = self.size[1] = 3
        prob = random.random()
        if prob < self.p:
            # flip all horizontal coords
            for unit in range(len(coords)):
                coords[unit][:,0] = self.size[0]-1-coords[unit][:,0] 
        return coords

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)






class RandomVerticalCoordsFlip(object):


    def __init__(self, size, p=0.5):
        super().__init__()
        self.p = p
        self.size = size

    def __call__(self, coords):

        prob = random.random()
        if prob < self.p:
            for unit in range(len(coords)):
                coords[unit][:,1] = self.size[1]-1-coords[unit][:,1] 
        return coords

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomTranspose(object):


    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img):

        if random.random() < self.p:
            return np.transpose(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)