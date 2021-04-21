from FrontDataset import *
from EraExtractors import *
from skimage.io import imsave
import argparse
import numpy as np

def setupDataset(data_location, label_location, variables):
    
    outsizes = (721, 1440)
    mapTypes = {"NA": ("NA", (90,-90.01), (-180,180), (-0.25,0.25)) }
    #outsizes = (160, 320)
    #mapTypes = {"NA": ("NA", (70,30), (-40,40), (-0.25,0.25)) }

    transforms = (None, None)

    myLevelRange = np.arange(105,138,2)
    #myLevelRange = np.arange(137,138,2)

    # train data set
    data_fold = data_location
    label_fold = label_location

    labelThickness = 1
    labelGroupingList = 'wcos'

    myLabelExtractor = None

    myEraExtractor = DerivativeFlippingAwareEraExtractor(variables, fliprate = 0.0, normType = None)

    data_set = WeatherFrontDataset(data_dir=data_fold, label_dir=label_fold, mapTypes=mapTypes, levelRange = myLevelRange, transform=transforms, outSize= outsizes, labelThickness = labelThickness, label_extractor = myLabelExtractor, era_extractor = myEraExtractor)

    print()
    print("Training Data...")
    print("Data-Location:", data_fold)
    print("Label-Location:", label_fold)
    print()

    return data_set


def calcStatistics(name, data, variable):
    mean = np.mean(data, axis = 0)
    mmax = np.max(data, axis = 0)
    mmin = np.min(data, axis = 0)
    var = np.var(data, axis = 0)
    print(mean.shape)
    print("{}: mean = {}, max = {}, min = {}, var = {}".format(variable, np.mean(mean), np.max(mmax), np.min(mmin), np.var(data)))
    mean.tofile("statimgs/mean/"+name+".bin")
    mmax.tofile("statimgs/max/"+name+".bin")
    mmin.tofile("statimgs/min/"+name+".bin")
    var.tofile("statimgs/var/"+name+".bin")


def calcStatisticsGlobal(name, data, variable):
    mean = np.mean(data)
    mmax = np.max(data)
    mmin = np.min(data)
    var = np.var(data)
    print(mean.shape)
    print("{}: mean = {}, max = {}, min = {}, var = {}".format(variable, (mean), (mmax), (mmin), (data)))
    mean.tofile("statimgs/mean/"+name+".bin")
    mmax.tofile("statimgs/max/"+name+".bin")
    mmin.tofile("statimgs/min/"+name+".bin")
    var.tofile("statimgs/var/"+name+".bin")

def saveStatistics(name, mean, mmax,mmin,var):
    imsave("statimgs/mean_"+name+".png", mean) 
    imsave("statimgs/var_"+name+".png", var)
    imsave("statimgs/max_"+name+".png", mmax)
    imsave("statimgs/min_"+name+".png", mmin)


def calcAllAtOnce(data_set, vars, num_samples, levels, latdim, londim):

    # use single levels due to restriction in memory
    data = np.zeros((len(vars), num_samples, 1, latdim, londim))
    monthmask = np.zeros(num_samples)
    yearmask = np.zeros(num_samples)
    hourmask = np.zeros(num_samples)
    years = [0]
    months = [x for x in range(12)]
    hours = [x for x in range(4)]

    tgtLevel = 0

    for sample in range(num_samples):
        idx = sample
        print("\rreading sample:",sample, "/", num_samples,"index:", idx,"/",num_samples, end='')
        
        smpl = data_set.__getitem__(idx)
        name = smpl[2]
        smpl = smpl[:2]
        year,month,hour = name[:4], name[4:6], name[9:11]
        for idx2,variable in enumerate(vars):
            #data[idx2, sample] = smpl[0][idx2*levels:(idx2+1)*levels].numpy()
            data[idx2, sample] = smpl[0][idx2*levels+tgtLevel].numpy()
            monthmask[sample] = int(month)-1
            yearmask[sample] = 0
            hourmask[sample] = int(hour)//6
    for idx,variable in enumerate(vars):
        # global average
        print(data.shape[1])
        print("overall")
        print("{}: mean = {}, max = {}, min = {}, var = {}".format(variable, np.mean(data[idx]), np.max(data[idx]), np.min(data[idx]), np.var(data[idx])))
        calcStatistics(str(variable)+"_l"+str(tgtLevel)+"_overall",data[idx], variable)
        for yearIdx in years:
            # yearly average
            yearSamples = yearmask==yearIdx
            #subdata = data[idx, yearSamples]
            #print("per year: ",yearSamples.sum())
            #print("{}: mean = {}, max = {}, min = {}, var = {}".format(variable, np.mean(subdata), np.max(subdata), np.min(subdata), np.var(subdata)))
            for monthIdx in months:
                # monthly average per year
                monthSamples = yearSamples * (monthmask==monthIdx)
                subdata = data[idx, monthSamples]
                print("\tper month per year: ",monthIdx, monthSamples.sum())
                if(monthSamples.sum()>0):
                    calcStatistics(str(variable)+"_l"+str(tgtLevel)+"_m"+str(monthIdx), subdata, variable)
                for hourIdx in hours:
                    # hourly average per month per year
                    hourSamples = monthSamples * (hourmask == hourIdx)
                    subdata = data[idx, hourSamples]
                    print("\t\tper hour per month per year: ", hourIdx, hourSamples.sum())
                    if(hourSamples.sum()>0):
                        calcStatistics(str(variable)+"_l"+str(tgtLevel)+"_m"+str(monthIdx)+"_h"+str(hourIdx), subdata, variable)
            for hourIdx in hours:
                # hourly average per year
                hourSamples = yearSamples * (hourmask == hourIdx)
                subdata = data[idx, hourSamples]
                print("\tper hour per year: ", hourIdx, hourSamples.sum())
                if(hourSamples.sum()>0):
                    calcStatistics(str(variable)+"_l"+str(tgtLevel)+"_h"+str(hourIdx), subdata, variable)



def calcEachIndividual(data_set, vars, num_samples, levels, latdim, londim):
    
    filenames = data_set.getNames()
    monthmasks = [[] for x in range(12)]
    hourmasks = [[] for x in range(4)]
    monthhourmasks = [[[] for x in range(4)] for y in range(12)]
    for idx, filename in enumerate(filenames):
        year,month,hour = filename[:4], filename[4:6], filename[9:11]
        monthmasks[int(month)-1].append(idx)
        hourmasks[int(hour)//6].append(idx)
        monthhourmasks[(int(month)-1)][(int(hour)//6)].append(idx)
    
    #for now just monthhoutmasks is possible
    for month in range(0,12):
        for hour in range(0,4):
            print("month:", month+1, "hour:", hour*6)
            data = np.zeros((len(vars), len(monthhourmasks[month][hour]), levels, latdim, londim))
            # Read all files to array
            for idx,fileIdx in enumerate(monthhourmasks[month][hour]):
                ns = len(monthhourmasks[month][hour])
                print("\rreading sample:",idx, "/", ns,"index:", fileIdx,"/",num_samples, end='')
                smpl = data_set.__getitem__(fileIdx)
                for idx2, var in enumerate(vars):
                    data[idx2, idx] = smpl[0][idx2*levels:(idx2+1)*levels].numpy()
            # Calc Statistics
            for idx, var in enumerate(vars):
                calcStatistics(str(var)+"_m"+str(month+1)+"_h"+str(hour*6),data[idx], var)

def calcYearlyStats(data_set, vars, num_samples, levels, latdim, londim):
    
    data = np.zeros((len(vars), levels, latdim, londim))
    vardata = np.zeros_like(data)
    for idx in range(num_samples):
        smpl = data_set[idx][0]
        for idx2,var in enumerate(vars):
            data[idx2] += smpl[idx2*levels:(idx2+1)*levels].numpy()
            vardata[idx2] += smpl[idx2*levels:(idx2+1)*levels].numpy()*smpl[idx2*levels:(idx2+1)*levels].numpy()
    data /= num_samples
    mean = np.mean(data, axis = (1,2,3))
    vardata /= num_samples
    var = vardata-mean*mean
    print(mean, var)
    mean.tofile("statimgs/min/global.bin")
    var.tofile("statimgs/var/global.bin")

    


if __name__ == "__main__":
    '''
    build a sample database and save data, label and the mapped images
    '''
    parser = argparse.ArgumentParser(description='FrontNet')
    parser.add_argument('--data', help='path to folder containing data')
    parser.add_argument('--label', type = str, default = None, help='path to folder containing label')
    parser.add_argument('--vars', nargs="+" , type = str, default = None, help='path to folder containing data')
    args = parser.parse_args()

    vars = args.vars

    data_set = setupDataset(args.data, args.label, vars)

    data_sample = data_set[0][0]

  
    levels, latdim, londim = data_sample.shape
    levels //= len(vars)
    print("levels:", levels, "with levelrange:", data_set.levelrange)
    print("lat pixel:", latdim, "with latrange", data_set.mapTypes["NA"][1])
    print("lon pixel:", londim, "with lonrange", data_set.mapTypes["NA"][2])

    num_samples = len(data_set)
    #calcAllAtOnce(data_set, vars, num_samples, levels, latdim, londim)
    #calcEachIndividual(data_set, vars, num_samples, levels, latdim, londim)
    calcYearlyStats(data_set, vars, num_samples, levels, latdim, londim)


