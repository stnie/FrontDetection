import numpy as np
from skimage.draw import line
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.ndimage import map_coordinates
import random

def extractHRFrontsGen(filename, classes, lonOff, latOff):
    fronts = [[] for _ in range(len(classes))]
    currentClass = ''
    for line in open(filename, 'r'):
        content = line.split()
        if(len(content)==0):
            currentClass = ''
            continue
        if(content[0] == '48HR'):
            break
        # If we encounter a no front class keyword of the format, reset the currentClass and go to next line
        if(content[0] in ['$$','TROF', 'LOWS', 'HIGHS']):
            currentClass = ''
            continue
        # if we encounter a front class keyword of the format, reset the currentClass and process the line
        if(content[0] in ["WARM", "COLD", "OCFNT", "STNRY"]):
            currentClass = ''
        for idx, className in enumerate(classes):
            if(content[0] == className):
                currentClass = className
                latCoords = np.zeros(len(content)-1)
                lonCoords = np.zeros(len(content)-1)
                # HR has no classification in intensity
                # csb Latitude is in degrees north
                # csv Longitude is in degrees west
                for idx2, coord in enumerate(content[1:]):
                    lat = int(coord[:3])/10 - latOff
                    lon = -int(coord[3:])/10 - lonOff
                    latCoords[idx2] = lat#round((latRes)//2-(1/latStep)*(lat))%(latRes)
                    lonCoords[idx2] = lon#round((1/lonStep)*(lon))%(lonRes)
                fronts[idx].append(latCoords)
                fronts[idx].append(lonCoords)
            # Old class continues
            elif(currentClass == className):
                latCoords = np.zeros(len(content)+1)
                lonCoords = np.zeros(len(content)+1)
                # set start at end of previous line to leave no gaps
                latCoords[0] = fronts[idx][-2][-1]
                lonCoords[0] = fronts[idx][-1][-1]
                # HR has no classification in intensity
                # csb Latitude is in degrees north
                # csv Longitude is in degrees west
                for idx2, coord in enumerate(content):
                    lat = int(coord[:3])/10 - latOff
                    lon = -int(coord[3:])/10 - lonOff
                    latCoords[idx2+1] = lat#round((latRes)//2-(1/latStep)*(lat))%latRes
                    lonCoords[idx2+1] = lon#round((1/lonStep)*(lon))%lonRes
                fronts[idx].append(latCoords)
                fronts[idx].append(lonCoords)
            
    return fronts



def degToRegularGrid(fronts, res):
    latRes = (np.abs(180/res[0])+1).astype(np.int32)
    lonRes = int(360/res[1])
    for type in fronts:
        for frontidx in range(0,len(type),2):
            for pairIdx in range(len(type[frontidx])):
                lat = type[frontidx][pairIdx]
                lon = type[frontidx+1][pairIdx]
                type[frontidx][pairIdx] = round((latRes)//2+(1/res[0])*(lat))%latRes
                type[frontidx+1][pairIdx] = round((1/res[1])*(lon))%lonRes
    return fronts

def extractFrontsSelfCreatedNoDuplicates(filename, classes, lonOff, latOff):
    fronts = [[] for x in range(len(classes))]
    for line in open(filename, 'r'):
        content = line.split()
        if(len(content)==0):
            continue
        for idx, className in enumerate(classes):
            if(content[0] == className):
                latCoords = []#np.zeros((len(content)-1)//2)
                lonCoords = []#np.zeros((len(content)-1)//2)

                # basis change such than lat ranges from 180 (bot) to 0 (top)
                # and lon ranges from 0 left to 360 right
                lastLat = -1
                lastLon = -1
                for idx2 in range(1,len(content),2):
                    lat = float(content[idx2][1:-1]) - latOff
                    lon = float(content[idx2+1][:-1]) - lonOff
                    newLat = lat#round(latRes//2-(1/latStep)*(lat))%latRes
                    newLon = lon#round((1/lonStep)*(lon))%lonRes
                    # Only extract a point if it is different from the previous (do not generate duplicates)
                    if(newLat != lastLat or newLon != lastLon):
                        lastLat = newLat
                        lastLon = newLon
                        latCoords.append(lastLat)
                        lonCoords.append(lastLon)
                fronts[idx].append(np.array(latCoords))
                fronts[idx].append(np.array(lonCoords))
    return fronts


def extractPolyLines(fronts, lonRes, latRes, thickness = 1):
    pls = np.zeros((latRes, lonRes, len(fronts)+1))
    pls[:,:,0] = np.ones((latRes,lonRes))
    # for each type of front detected
    for idx, ft in enumerate(fronts):
        # for each individual front of the given type
        for instance in range(0,len(ft),2):
            latCoords = ft[instance]
            lonCoords = ft[instance+1]
            for idx2 in range(len(lonCoords)-1):
                possWays = np.array([np.linalg.norm(lonCoords[idx2]-lonCoords[idx2+1]), np.linalg.norm(lonCoords[idx2]-(lonCoords[idx2+1]-lonRes)), np.linalg.norm(lonCoords[idx2]-lonRes-lonCoords[idx2+1])])
                pos = np.argmin(possWays)
                if(pos == 1):
                    lonCoords[idx2+1] -= lonRes
                elif(pos == 2):
                    lonCoords[idx2] -= lonRes
                rr, cc = line(int(latCoords[idx2]), int(lonCoords[idx2]), int(latCoords[idx2+1]), int(lonCoords[idx2+1]) )
                
                for lt in range(-(thickness//2),thickness//2+1):
                    pls[rr%latRes,(cc+lt)%lonRes, idx+1] = 1
                    pls[(rr+lt)%latRes,cc%lonRes, idx+1] = 1
                    pls[rr%latRes,(cc+lt)%lonRes, 0] = 0
                    pls[(rr+lt)%latRes,cc%lonRes, 0] = 0
    return pls

def extractFlatPolyLines(fronts, lonRes, latRes, thickness = 1):
    image = np.zeros((latRes, lonRes, 1))
    # for each type of front detected
    for idx, ft in enumerate(fronts):
        # for each individual front of the given type
        for instance in range(0,len(ft),2):
            latCoords = ft[instance]
            lonCoords = ft[instance+1]
            # for each coordinate pair of an instance
            for idx2 in range(len(lonCoords)-1):
                possWays = np.array([np.linalg.norm(lonCoords[idx2]-lonCoords[idx2+1]), np.linalg.norm(lonCoords[idx2]-(lonCoords[idx2+1]-lonRes)), np.linalg.norm(lonCoords[idx2]-lonRes-lonCoords[idx2+1])])
                pos = np.argmin(possWays)
                if(pos == 1):
                    lonCoords[idx2+1] -= lonRes
                elif(pos == 2):
                    lonCoords[idx2] -= lonRes
                # extract line from [lat,lon] to [lat,lon]
                rr, cc = line(int(latCoords[idx2]), int(lonCoords[idx2]), int(latCoords[idx2+1]), int(lonCoords[idx2+1]) )
                # idx + 1 as the zero label is used to determine the background
                sigma = 3
                if(sigma > 0):
                    norm_fac = 1/(sigma*np.sqrt(2*np.pi))
                    sigma2 = sigma*sigma
                    for lt in range(-(thickness//2),1):
                        lt2 = lt*lt
                        value = norm_fac * np.exp(-0.5*lt2/sigma2)
                        print("image value is ", value)
                        image[rr,(cc+lt)%lonRes,0] = value
                        image[(rr+lt)%latRes,cc,0] = value
                        image[rr,(cc-lt)%lonRes,0] = value
                        image[(rr-lt)%latRes,cc,0] = value
                else:
                    for lt in range(-(thickness//2),thickness//2+1):
                        image[rr,(cc+lt)%lonRes,0] = idx+1
                        image[(rr+lt)%latRes,cc,0] = idx+1
                    
    return image


def extractLines(fronts, lonRes, latRes):
    myLines = []
    # for each type of front detected
    for idx, ft in enumerate(fronts):
        myLines.append([])
        # for each individual front of the given type
        for instance in range(0,len(ft),2):
            latCoords = ft[instance]
            lonCoords = ft[instance+1]
            # for each coordinate pair of an instance
            for idx2 in range(len(lonCoords)-1):
                possWays = np.array([np.linalg.norm(lonCoords[idx2]-lonCoords[idx2+1]), np.linalg.norm(lonCoords[idx2]-(lonCoords[idx2+1]-lonRes)), np.linalg.norm(lonCoords[idx2]-lonRes-lonCoords[idx2+1])])
                pos = np.argmin(possWays)
                if(pos == 1):
                    lonCoords[idx2+1] -= lonRes
                elif(pos == 2):
                    lonCoords[idx2] -= lonRes
                # extract line from [lat,lon] to [lat,lon]
                rr, cc = line(int(latCoords[idx2]), int(lonCoords[idx2]), int(latCoords[idx2+1]), int(lonCoords[idx2+1]) )
                myLines[idx].append((rr,cc))
    return myLines

def drawOffsettedLines(image, line, value, thickness, offset , lonRes, latRes):
    rr, cc = line
    for lt in range(-(thickness//2),thickness//2+1):
        image[(rr+offset[0])%latRes,(cc+lt+offset[1])%lonRes,0] = value
        image[(rr+lt+offset[0])%latRes,((cc+offset[1])%lonRes),0] = value

def cropToRange(image, latRange, lonRange, res):
    latRange = (90-np.arange(latRange[0], latRange[1], res[0]))/np.abs(res[0])
    lonRange = np.arange(lonRange[0], lonRange[1], res[1])/np.abs(res[1])
    image = image[latRange.astype(np.int32),:,:]
    image = image[:,lonRange.astype(np.int32),:]
    return image


class extractFlatPolyLinesInRange():
    def __init__ (self, labelGrouping = None, thickness = 1, maxOff = (0,0)):
        self.labelGrouping = labelGrouping
        self.fieldToNum = {"w":1,"c":2,"o":3,"s":4}
        if(self.labelGrouping is None):
            self.labelGrouping = "wcos"
        groupStrings = self.labelGrouping.split(',')
        self.groups = [[self.fieldToNum[member] for member in group] for group in groupStrings]

        #print("fpl",self.labelGrouping, self.groups)
        self.thickness = thickness
        self.maxOff = maxOff
    def __call__(self,fronts, latRange, lonRange, res):
        latRes = (np.abs(180/res[0])+1).astype(np.int32)
        lonRes = int(360/res[1])
        # Groupings of different frontal types
        ftypes = len(self.groups)
        image = np.zeros((latRes, lonRes, 1))
        alllines = extractLines(fronts, lonRes, latRes)
        # draw the lines
        for idx, lines in enumerate(alllines,1):
            for grpidx, group in enumerate(self.groups,1):
                if idx in group:
                    tgtGrp = grpidx
            for line in lines:
                drawOffsettedLines(image, line, tgtGrp, self.thickness, self.maxOff, lonRes, latRes)
        # crop the image
        image = cropToRange(image, latRange, lonRange, res)
        return image



class extractCoordsInRange():
    def __init__(self, labelGrouping = None):
        self.labelGrouping = labelGrouping
        self.fieldToNum = {"w":1,"c":2,"o":3,"s":4}
        if(self.labelGrouping is None):
            self.labelGrouping = "wcos"
        groupStrings = self.labelGrouping.split(',')
        self.groups = [[self.fieldToNum[member] for member in group] for group in groupStrings]
        self.thickness = 1
        self.maxOff = (0,0)
    
    def __call__(self, fronts, latRange, lonRange, res):
        latRes = (np.abs(180/res[0])+1).astype(np.int32)
        lonRes = int(360/res[1])
        # Groupings of different frontal types
        ftypes = len(self.groups)
        allGroupedFronts = [[] for _ in range(ftypes)]
       
        for grpidx, group in enumerate(self.groups):
            for member in group:
                allGroupedFronts[grpidx] += fronts[member-1]
        # alls fronts are now grouped
        # Now: Merge connected Fronts of the same type
        groupedFronts = [[] for _ in range(ftypes)]
        closeDistance = 3
        for grpIdx in range(len(self.groups)):
            validList = [True for _ in range(len(allGroupedFronts[grpIdx]))]
            for i in range(0,len(allGroupedFronts[grpIdx]),2):
                istart = np.array(allGroupedFronts[grpIdx][i:i+2])[:,0]
                iend = np.array(allGroupedFronts[grpIdx][i:i+2])[:,-1]
                # empty ranges should be removed
                if(np.all(istart == iend)):
                    validList[i] = False
                    validList[i+1] = False
                    continue
                for j in range(i+2,len(allGroupedFronts[grpIdx]),2):
                    jstart = np.array(allGroupedFronts[grpIdx][j:j+2])[:,0]
                    jend = np.array(allGroupedFronts[grpIdx][j:j+2])[:,-1]
                    if(np.all(jstart == jend)):
                        continue
                    # connection type 1
                    if(np.linalg.norm(istart-jstart)<closeDistance):
                        allGroupedFronts[grpIdx][j] = np.concatenate((np.flip(allGroupedFronts[grpIdx][i], axis=0), allGroupedFronts[grpIdx][j]), axis = 0)
                        allGroupedFronts[grpIdx][j+1] = np.concatenate((np.flip(allGroupedFronts[grpIdx][i+1], axis=0), allGroupedFronts[grpIdx][j+1]), axis = 0)
                        validList[i] = False
                        validList[i+1] = False
                        break
                    elif(np.linalg.norm(istart - jend)< closeDistance):
                        allGroupedFronts[grpIdx][j] = np.concatenate((allGroupedFronts[grpIdx][j], allGroupedFronts[grpIdx][i]), axis = 0)
                        allGroupedFronts[grpIdx][j+1] = np.concatenate((allGroupedFronts[grpIdx][j+1], allGroupedFronts[grpIdx][i+1]), axis = 0)
                        validList[i] = False
                        validList[i+1] = False
                        break
                    elif(np.linalg.norm(iend - jstart)< closeDistance):
                        allGroupedFronts[grpIdx][j] =  np.concatenate((allGroupedFronts[grpIdx][i], allGroupedFronts[grpIdx][j]), axis = 0)
                        allGroupedFronts[grpIdx][j+1] =  np.concatenate((allGroupedFronts[grpIdx][i+1], allGroupedFronts[grpIdx][j+1]), axis = 0)
                        validList[i] = False
                        validList[i+1] = False
                        break
                    elif(np.linalg.norm(iend - jend)< closeDistance):
                        allGroupedFronts[grpIdx][j] =  np.concatenate((allGroupedFronts[grpIdx][j], np.flip(allGroupedFronts[grpIdx][i], axis = 0)), axis = 0)
                        allGroupedFronts[grpIdx][j+1] =  np.concatenate((allGroupedFronts[grpIdx][j+1], np.flip(allGroupedFronts[grpIdx][i+1], axis = 0)), axis = 0)
                        validList[i] = False
                        validList[i+1] = False
                        break
            for i in range(len(validList)):
                if(validList[i]):
                    groupedFronts[grpIdx].append(allGroupedFronts[grpIdx][i])

        # groupedFronts now hold a concatenation of same Type fronts where two ends were in the same spot
        # Now remove all lines outside the target range
        # We define a line outside if both vertices are outside the inspected window
        # If only one vertex is outside, we move the vertex to the next border pixel along the line 

        allGroups = []
        # transform from degree range into pixel range (relative to the whole grid)
        latRange = (((90-latRange[0])/np.abs(res[0]))%latRes,((90-latRange[1])/np.abs(res[0]))%latRes)
        lonOff = 0
        if(lonRange[0]<0 and lonRange[1]>0):
            lonOff = -180
        lonRange = (((lonRange[0]-lonOff)/res[1])%lonRes, ((lonRange[1]-lonOff)/res[1])%lonRes)
        for grpidx, frontgroup in enumerate(groupedFronts):
            newgroup = []
            for instance in range(0,len(frontgroup),2):
                thisLats = []
                thisLons = []
                latCoords = np.array(frontgroup[instance])
                lonCoords = (np.array(frontgroup[instance+1])-lonOff/res[1])%lonRes
                previsin = False
                inlat = latCoords[0] < latRange[1] and latCoords[0] > latRange[0]
                inlon = lonCoords[0] < lonRange[1] and lonCoords[0] > lonRange[0]
                isin = inlat and inlon
                point = 0
                for point in range(1,len(lonCoords)):
                    #checkNext
                    nextinlat = latCoords[point] < latRange[1] and latCoords[point] > latRange[0]
                    nextinlon = lonCoords[point] < lonRange[1] and lonCoords[point] > lonRange[0]
                    nextisin = nextinlat and nextinlon
                    if(previsin or isin or nextisin):
                        thisLats.append(latCoords[point-1])
                        thisLons.append(lonCoords[point-1])
                    else:
                        # The vertex is completely outside the region
                        # We start a new Front segment
                        if(len(thisLons)>0):
                            newgroup.append(thisLats)
                            newgroup.append(thisLons)
                            thisLats = []
                            thisLons = []
                    previsin = isin
                    isin = nextisin
                if(previsin or isin):
                    thisLats.append(latCoords[point])
                    thisLons.append(lonCoords[point])
                if(len(thisLons)>0):
                    newgroup.append(thisLats)
                    newgroup.append(thisLons)
                
            allGroups.append(newgroup)
        # now all filtered fronts should be within the group
        # We now return the coordinates relative to the origin of the extracted region
        return [[np.array([(np.array(allGroups[group][2*x])-latRange[0]), (np.array(allGroups[group][2*x+1])-lonRange[0])]).transpose().astype(np.int32) for x in range(len(allGroups[group])//2)] for group in range(len(allGroups))]

class extractStackedPolyLinesInRangeAsSignedDistance():
    def __init__ (self, labelGrouping = None, thickness = 1, maxOff = (0,0)):
        self.labelGrouping = labelGrouping
        self.fieldToNum = {"w":1,"c":2,"o":3,"s":4}
        if(self.labelGrouping is None):
            self.labelGrouping = "wcos"
        groupStrings = self.labelGrouping.split(',')
        self.groups = [[self.fieldToNum[member] for member in group] for group in groupStrings]
        
        #print(self.labelGrouping, self.groups)
        self.thickness = thickness
        self.maxOff = maxOff
        
    def __call__(self,fronts, latRange, lonRange, res):
        latRes = (np.abs(180/res[0])+1).astype(np.int32)
        lonRes = int(360/res[1])
        # Groupings of different frontal types
        ftypes = len(self.groups)
        image = np.zeros((latRes, lonRes, ftypes))
        alllines = extractLines(fronts, lonRes, latRes)
        # draw the lines
        for idx, lines in enumerate(alllines,1):
            for grpidx, group in enumerate(self.groups):
                if idx in group:
                    tgtGrp = grpidx
                    for line in lines:
                        # A random offset to the line segment. Reduce the personal bias of the label creator
                        drawOffsettedLines(image[:,:,tgtGrp:tgtGrp+1], line, 1, 1, self.maxOff, lonRes, latRes)
        if(self.thickness == 1):
            image = cropToRange(image, latRange, lonRange, res)
            return image
        else:
            # flip 1 and 0, so that label is background
            image = (image==0)*1
            # if we cross the 0 Degree point we need to shift before calculating the distance transform to not get border effects
            if(lonRange[0]< 0 and lonRange[1] >= 0):
                image = np.roll(image, lonRes//2, axis = 1)
                for ft in range(ftypes):
                    image[:,:,ft] = ndimage.distance_transform_edt(image[:,:,ft], return_distances = True, return_indices = False)
                image = np.roll(image, -lonRes//2, axis = 1)
            else:
                for ft in range(ftypes):
                    image[:,:,ft] = ndimage.distance_transform_edt(image[:,:,ft], return_distances = True, return_indices = False)
            # crop the image
            image = cropToRange(image, latRange, lonRange, res)
            # clip to range [0, thickness]
            image = np.clip(image, 0, self.thickness)
            image = 1-image/self.thickness
            return image



class DefaultFrontLabelExtractor():
    # Determine which lines should be extracted and with which size
    def __init__(self, imageCreator = extractFlatPolyLinesInRange):
        self.imageCreator = imageCreator

    def __call__(self, filename, latrange, lonrange, res, labtype):
        if isinstance(filename, list):
            print("Currently not correct!")
            exit(1)
            allFronts = []
            for idx, filenam in enumerate(filename):
                allFronts.append(this(filenam, latrange, lonrange, res, labtype))
            return np.array(allFronts)

        # Need to separate labtype, because of slightly different file formats 
        fronts = self.getCoordinates(filename, labtype, res)

        # transform coordinates into lines
        tmp = self.imageCreator(fronts, latrange, lonrange, res)
        
        return tmp
    
    def getCoordinates(self, filename, labtype, res):
        if(labtype in ["NA", "NT", ""]): 
            available_types = ["warm", "cold", "occ", "stnry"]
            degFronts = extractFrontsSelfCreatedNoDuplicates(filename, available_types,0,0)
        elif(labtype in ["hires"]):
            available_types = ["WARM","COLD","OCFNT", "STNRY"]
            degFronts = extractHRFrontsGen(filename, available_types, 0, 0)
        else:
            print("invalid Labtype: ", labtype)
            exit(1)
        return degToRegularGrid(degFronts, res)


