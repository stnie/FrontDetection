import numpy as np
import sys
from skimage.io import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from skimage.morphology import binary_dilation
from scipy.stats import binom
from scipy.ndimage import distance_transform_edt
import pandas as pd

def getRatioWithinRange(mixed_event, base_event, name=None, filter = None):
    if(name is None):
        name = "global"
    if(filter is None):
        filter = []
    me = cropToRegion(mixed_event, name, filter)
    be = cropToRegion(base_event, name, filter)
    zpos = np.nonzero(be)
    c = me[zpos]/(be[zpos])
    ratio = np.sum(c)/len(zpos[0])
    return (name+"_"+str(filter), ratio)


def getAllRatios(mixed_event, base_event):
    pregions = ["global", "northern hemisphere", "southern hemisphere", "midlat", "northern midlat",
    "southern midlat", "tropics"]
    filter =[]
    regions = []
    for r in pregions:
        regions += [r]*5
        filter += [[], "no mountain", "sea", "land", ["land","no mountain"]]
    ratios = []
    names = []
    for idx in range(len(regions)):
        name, count = getRatioWithinRange(mixed_event, base_event, regions[idx], filter[idx])
        ratios.append(count)
        names.append(name)
    return ratios, names

def plotXY(x_event, y_event, num_bins, region, savename):
    x_e = cropToRegion(x_event, region)
    y_e = cropToRegion(y_event, region)
    xtarg = np.linspace(0,1,num_bins)
    ytarg = np.zeros_like(xtarg)
    for x in range(len(xtarg)):
        if(x==0):
            pos = np.nonzero((x_e <= xtarg[x]))
        else:
            pos = np.nonzero((x_e <= xtarg[x]) * (x_e>xtarg[x-1]))
        if(len(pos[0])>0):
            ytarg[x] = np.mean(y_e[pos])
        else:
            ytarg[x] = np.NaN
    ytarg[-1] = np.NaN
    plt.xlim(0,0.6)
    plt.ylim(0.005,0.015)
    plt.scatter(xtarg, ytarg)
    plt.xlabel("p(front)")
    plt.ylabel("p(extreme precip)")
    plt.title(savename)
    plt.savefig("xyplot_"+savename+".png")
    plt.gcf().clear()


def plotBoxPlt(x_event, y_event, y_event_div, num_bins, region, filter, savename, pct01, pct99):
    x_e = cropToRegion(x_event, region, filter)
    y_e = cropToRegion(y_event, region, filter)
    y_ed = cropToRegion(y_event_div, region, filter)
    pos = np.nonzero(y_ed)
    x_e = x_e[pos]
    y_e = y_e[pos]/y_ed[pos]
    xtarg = np.linspace(0,1,num_bins)
    ytarg = []
    minCount = 0
    for x in range(len(xtarg)):
        if(x==0):
            pos = np.nonzero((x_e <= xtarg[x]))
        else:
            pos = np.nonzero((x_e <= xtarg[x]) * (x_e>xtarg[x-1]))
        if(len(pos[0])>minCount):
            ytarg.append(y_e[pos])
        else:
            ytarg.append(np.NaN)
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.plot(xtarg[:num_bins])
    ax.plot(xtarg*pct01[1]+pct01[0])
    ax.plot(xtarg*pct99[1]+pct99[0])
    ax.boxplot(ytarg, positions=np.arange(num_bins), showmeans=True)
    plt.xticks(np.arange(num_bins), (xtarg*100).astype(np.int32))
    
    plt.ylim(-0.05,1.05)
    plt.xlabel("$P_{a(fr)}$")
    plt.ylabel("$R_1$")
    savename += "_{}".format(region)
    for filt in filter:
        savename += "_{}".format(filt)
    plt.legend(["Identity", "1st percentile", "99th percentile"])
    front_type = savename.split("/")[-1].split("_")[1]
    if(front_type == "stnry"):
        front_type = "stationary fronts"
    elif(front_type == "occ"):
        front_type = "occlusions"
    elif(front_type == "all"):
        front_type = "any front"
    else:
        front_type += " fronts"
    plt.title("Proportion of Extreme Precipitation Events associated with "+front_type)
    fig.savefig(savename+".png")
    plt.close(fig)



def createDensityDiff(mixed_events, base_event_dependant, base_event_independent, region, filter,savename):
    me = cropToRegion(mixed_events, region, filter)
    bed = cropToRegion(base_event_dependant, region, filter)
    bei = cropToRegion(base_event_independent, region, filter)
    pos = np.nonzero(bed)
    avgF = (me[pos]/(bed[pos]))
    avgT = (bei[pos])
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.hist(avgF, bins=np.linspace(0,0.1,100), density = True)
    ax.hist(avgT, bins=np.linspace(0,0.1,100), density = True)
    savename += "_{}".format(region)
    for filt in filter:
        savename += "_{}".format(filt)
    plt.title(savename)
    plt.legend(["$N_{front_and_extreme}/N_{front}$", "$N_{extreme}/N$"])
    fig.savefig(savename+".png")
    plt.close(fig)

# these are inclusive ranges
def getLatRange(tgt_region):
    if(tgt_region == "northern midlat"):
        return[(100,221)]
    elif(tgt_region == "southern midlat"):
        return[(460,581)]
    elif(tgt_region == "midlat"):
        return[(101,221),(460,581)]
    elif(tgt_region == "tropics"):
        return[(220,461)]
    elif(tgt_region == "northern hemisphere"):
        return[(100, 341)]
    elif(tgt_region == "southern hemisphere"):
        return[(340, 581)]
    elif(tgt_region == "global"):
        return[(100, 581)]
    

def cropToRegion(data, tgt_region, filter=[]):
    remove_mountains = False
    remove_non_mountains = False
    remove_land = False
    remove_sea = False
    if("no mountain" in filter):
        remove_mountains = True
    if("only mountain" in filter):
        remove_non_mountains = True
    if("land" in filter):
        remove_sea = True
    if("sea" in filter):
        remove_land = True
    ranges = getLatRange(tgt_region)
    loc_h = []
    loc_lsm = []
    loc_d = []
    for fr,to in ranges:
        loc_h.append(height[fr:to])
        loc_lsm.append(land_sea[fr:to])
        loc_d.append(data[fr:to])
    loc_h = np.concatenate(loc_h)
    loc_lsm = np.concatenate(loc_lsm)
    loc_d = np.concatenate(loc_d)
    filt = np.ones_like(loc_d)
    if(remove_mountains):
        filt *= loc_h
    elif(remove_non_mountains):
        filt *= (1-loc_h)
    if(remove_sea):
        filt *= loc_lsm
    elif(remove_land):
        filt *= 1-loc_lsm
    filteredPos = np.nonzero(filt)
    return loc_d[filteredPos]
    


def saveImg(image, savename, base_low = None, base_up = None):
    if(not base_low is None):
        pos = np.nonzero((image <= base_up) * (image >= base_low))
        image[pos] = np.NaN
    fig, ax = plt.subplots(1,1, figsize=(10,5),subplot_kw={'projection':ccrs.PlateCarree()})
    ax.set_global()
    ax.coastlines()
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    globim = np.zeros((480, 1440))
    globim[:,20:-20] = image[100:-100,:]
    globim[:,:20] = np.NaN
    globim[:,-20:] = np.NaN
    cmap = plt.get_cmap("RdYlBu", 10)
    im = ax.imshow(globim, extent = (-175,175,-60,60), cmap = cmap)
    axpos = ax.get_position()
    pos_x = axpos.x0 #+ 0.25*axpos.width
    pos_y = axpos.y0+axpos.height + 0.025
    cax_width = axpos.width
    cax_height = 0.2
    pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])
    cb = fig.colorbar(im, orientation="horizontal", ticks=np.arange(0,1.1,0.1), cax=pos_cax)
    fig.savefig(savename+".png")
    plt.close(fig)
    
    
def saveImgRegion(image, region, filter, savename, base_low = None, base_up = None, pct_filter_image = None, tickstep = 0.1):
    noValFilt = np.isnan(image)*1.0
    if(not base_low is None):
        if(pct_filter_image is None):
            pos = np.nonzero((image <= base_up) * (image >= base_low))
            image[pos] = np.NaN
        else:
            pos = np.nonzero((pct_filter_image <= base_up) * (pct_filter_image >= base_low))
            image[pos] = np.NaN
    fig, ax = plt.subplots(1,1, figsize=(9,4),subplot_kw={'projection':ccrs.PlateCarree()})
    ax.set_global()
    ax.coastlines()
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    globim = np.zeros((481, 1400))
    regions = getLatRange(region)
    fr, to = regions[0]
    # special crops require special solutions
    if(region == "midlat"):
        fr1, to1 = getLatRange("northern midlat")[0]
        globim[fr1-100:to1-100] = image[fr1:to1,:]
        fr2, to2 = getLatRange("southern midlat")[0]
        globim[fr2-100:to2-100] = image[fr2:to2,:]
        print(fr1, to1, fr2, to2)
        globim[to1-100+1:fr2-100] = np.NaN
    else:
        # default crop
        globim[fr-100:to-100] = image[fr:to,:]
        globim[:fr-100] = np.NaN
        globim[to-100:] = np.NaN
    if("no mountain" in filter):
        globim[np.nonzero(1-height[100:581])] = np.NaN
        noValFilt[np.nonzero(1-height)] = 0.5
    if("land" in filter):
        globim[np.nonzero(1-land_sea[100:581])] = np.NaN
    elif("sea" in filter):
        globim[np.nonzero(land_sea[100:581])] = np.NaN
    ax.imshow(1-noValFilt[100:581], extent = (-175,175,-60,60), cmap = 'gray',vmin=0,vmax=1, alpha=0.7)
    cmap = plt.get_cmap("RdYlBu", 10)
    im = ax.imshow(globim, extent = (-175,175,-60,60), cmap = cmap, vmin = 0, vmax = 10*tickstep)
    ax.set_extent([-175, 175, -60, 60], crs=ccrs.PlateCarree())
    axpos = ax.get_position()
    pos_x = axpos.x0 #+ 0.25*axpos.width
    pos_y = axpos.y0 - 0.1
    cax_width = axpos.width
    cax_height = 0.05
    pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])
    cb = fig.colorbar(im, orientation="horizontal", ticks=np.arange(0,11*tickstep,tickstep), cax = pos_cax)
    front_type = savename.split("/")[-1].split("_")[-2]
    if(front_type == "stnry"):
        front_type = "stationary fronts"
    elif(front_type == "occ"):
        front_type = "occlusions"
    elif(front_type == "all"):
        front_type = "any front"
    else:
        front_type += " fronts"
    base_type = "Extreme Precipitation"
    if(savename.split("/")[-1].split("_")[-3] == "fronts"):
        tmp = base_type
        base_type = front_type
        front_type = tmp
    ax.set_title("Proportion of {} Associated with {}".format(base_type, front_type))
    
    fig.savefig(savename+".png")
    plt.close(fig)

def getFrontalCount(data, region, filter=[]):
    result = []
    for i in range(data.shape[0]):
        rem_dat = cropToRegion(data[i], region, filter)
        result.append(np.sum(rem_dat))
    return ("{}_{}".format(region,filter), result)

def getFrontCountInRegion(fronts):
    pregions = ["global", "northern hemisphere", "southern hemisphere", "midlat", "northern midlat",
    "southern midlat", "tropics"]
    filter =[]
    regions = []
    for r in pregions:
        regions += [r]*5
        filter += [[], "no mountain", "sea", "land", ["land","no mountain"]]
    ratios = []
    names = []
    for idx in range(len(regions)):
        name, count = getFrontalCount(fronts, regions[idx], filter[idx])
        ratios.append(count)
        names.append(name)
    return ratios, names

def getPercentileInformation(percentileName, season):
    tgtFile = percentileName+"_"+season+"_"
    pct01 = np.zeros((5,2))
    pct50 = np.zeros_like(pct01)
    pct99 = np.zeros_like(pct01)
    for ft in range(5):
        with open(tgtFile+str(ft)+".txt", "r") as f:
            c = 0
            for line in f:
                if("pct" in line):
                    if c == ft:
                        pct01[ft] = (float(line.split()[1][1:]),float(line.split()[2][:-1]))
                        line = next(f)
                        pct50[ft] = (float(line.split()[1][1:]),float(line.split()[2][:-1]))
                        line = next(f)
                        pct99[ft] = (float(line.split()[1][1:]),float(line.split()[2][:-1]))
                    else:
                        next(f)
                        next(f)
                    c+=1
    return pct01, pct99


event_fold = sys.argv[1]
res_fold = sys.argv[2]
num_fronts = int(sys.argv[3])
season = sys.argv[4]
assert(season in ["djf", "mam", "jja", "son", "all"])
res_fold = os.path.join(res_fold, season)
if not os.path.isdir(res_fold):
    os.mkdir(res_fold)
    print("created Folder {}".format(res_fold))


percentileName = os.path.join(sys.argv[5],"myRandSampResults")



front_events_name = os.path.join(event_fold,"{}_{}.bin".format("front_extreme_events",season))
total_events_name = os.path.join(event_fold,"{}_{}.bin".format("extreme_events",season))
fronts_name = os.path.join(event_fold, "{}_{}.bin".format("fronts",season))

rev_front_events_name = os.path.join(event_fold+"_reversed","{}_{}.bin".format("front_extreme_events",season))
rev_fronts_name = os.path.join(event_fold+"_reversed", "{}_{}.bin".format("fronts",season))


timesteps = 1464
front_events = np.fromfile(front_events_name).reshape(-1,num_fronts,680,1400)
total_events = np.fromfile(total_events_name).reshape(-1,680,1400)
fronts = np.fromfile(fronts_name).reshape(-1, num_fronts, 680, 1400)

front_events_reversed = np.fromfile(rev_front_events_name).reshape(-1,num_fronts,680,1400)
fronts_reversed = np.fromfile(rev_fronts_name).reshape(-1, num_fronts, 680, 1400)

assert(fronts.shape[0] == total_events.shape[0])
assert(fronts.shape[0] == front_events.shape[0])
print(fronts.shape)
if(season in ["djf", "mam", "jja", "son"]):
    assert(fronts.shape[0] == 3)
elif(season == "all"):
    assert(fronts.shape[0] == 12)


dpm = np.array([31,29,31,30,31,30,31,31,30,31,30,31])

timesteps = 24*np.sum(dpm)
front_events = (np.sum(front_events, axis=0, keepdims =True))/timesteps
total_events = (np.sum(total_events, axis=0, keepdims =True))/timesteps
fronts = (np.sum(fronts, axis=0, keepdims =True))/timesteps
fronts_reversed = (np.sum(fronts_reversed, axis=0, keepdims =True))/timesteps
front_events_reversed = (np.sum(front_events_reversed, axis=0, keepdims =True))/timesteps

#offset, factor
pct01_par, pct99_par = getPercentileInformation(percentileName, season)

pct01 = np.zeros_like(fronts)
pct99 = np.zeros_like(fronts)
for ft in range(5):
    pct01[0,ft] = fronts[0,ft]*pct01_par[ft,1]+ pct01_par[ft,0]
    pct99[0,ft] = fronts[0,ft]*pct99_par[ft,1]+ pct99_par[ft,0]

ftypes = ["all", "warm", "cold", "occ", "stnry"]
month = ["jan","feb","mar","apr","may", "jun","jul","aug","sep","oct","nov","dec"]
month = ["djf", "mam", "jja", "son"]


### THESE WILL BE USED GLOBALLY
geopot = np.fromfile("z.bin").reshape(721,1440)
geopot = np.roll(geopot / 9.80665, 720, axis=1)
land_sea_map = np.fromfile("lsm.bin").reshape(721,1440)
land_sea_map = np.roll(land_sea_map, 720, axis=1)
# This will be used by the mapping functions

height = geopot[20:-21,20:-20]<2000
height = distance_transform_edt(height)>5

land_sea = land_sea_map[20:-21,20:-20] > 0.0


allratios = []
allratios_reversed = []
frontcount = []
countnames = []
rationames = []
for m in range(front_events.shape[0]):
    print("season:",season)
    monthratio = []
    monthratio_reversed = []
    per_month_front_events = front_events[m]
    per_month_total_events = total_events[m]
    per_month_fronts = fronts[m]
    counts, names = getFrontCountInRegion(fronts_reversed[m])
    frontcount.append(counts)
    countnames.append(names)
    for f in range(front_events.shape[1]):
        print("frontal type:", ftypes[f])
        filename = ftypes[f]+"_"+season
        local_front_events = per_month_front_events[f]
        local_fronts = per_month_fronts[f]
        local_events = per_month_total_events
        ratios, names = (getAllRatios(local_front_events, local_events))
        ratios2, names2 = (getAllRatios(front_events_reversed[m,f], fronts_reversed[m,f]))
        monthratio.append(ratios)
        monthratio_reversed.append(ratios2)
        if(len(rationames) == 0):
            rationames.append(names)
        createDensityDiff(local_front_events,local_fronts, local_events, "midlat", ["no mountain"], os.path.join(res_fold,"density_"+filename))
        plotBoxPlt(local_fronts, local_front_events, local_events, 21, "midlat", ["no mountain"], os.path.join(res_fold,"boxplot_"+filename), pct01_par[f], pct99_par[f])
        createDensityDiff(local_front_events,local_fronts, local_events, "tropics", ["no mountain", "sea"], os.path.join(res_fold,"density_"+filename))
        plotBoxPlt(local_fronts, local_front_events, local_events, 21, "tropics", ["no mountain", "sea"], os.path.join(res_fold,"boxplot_"+filename), pct01_par[f], pct99_par[f])
        createDensityDiff(local_front_events,local_fronts, local_events, "midlat", ["no mountain", "land"], os.path.join(res_fold,"density_"+filename))
        plotBoxPlt(local_fronts, local_front_events, local_events, 21, "midlat", ["no mountain", "land"], os.path.join(res_fold,"boxplot_"+filename), pct01_par[f], pct99_par[f])
        epos = np.nonzero(local_events==0)
        local_front_events[epos] = np.NaN
        local_events[epos] = np.NaN
        fpos = np.nonzero(fronts_reversed[m,f]==0)
        front_events_reversed[m,f][fpos] = np.NaN
        fronts_reversed[m,f][fpos] = np.NaN
        saveImgRegion(local_front_events/(local_events), "global", ["no mountain"], os.path.join(res_fold,"prop_events_"+filename), pct01[m,0], pct99[m,0],per_month_front_events[0]/(local_events))
        saveImgRegion((front_events_reversed[m,f]/(fronts_reversed[m,f])), "global", ["no mountain"], os.path.join(res_fold,"prop_fronts_"+filename), pct01[m,0] , pct99[m,0],  per_month_front_events[0]/(local_events), 0.1)
        local_events[epos] = 0
    allratios.append(monthratio)
    allratios_reversed.append(monthratio_reversed)

allcountsarr = np.array(frontcount[0])
allcountsrel = allcountsarr.transpose()
allcountsrel = (allcountsrel / allcountsrel[0]).transpose()
# create data frame for the counts
count_frame = pd.DataFrame(allcountsarr, index = countnames, columns = ftypes)
count_frame_rel = pd.DataFrame(allcountsrel, index = countnames, columns = ftypes)

# Create data frame for the ratios
allvals = np.array(allratios[0]).transpose()
allvallsarr = np.array(allvals)
ratio_frame = pd.DataFrame(allvallsarr, index = rationames, columns= ftypes)

allvals_reversed = np.array(allratios_reversed[0]).transpose()
allvallsarr_reversed = np.array(allvals_reversed)
ratio_frame_reversed = pd.DataFrame(allvallsarr_reversed, index = rationames, columns= ftypes)

relative_ratio = pd.DataFrame(allvallsarr/allcountsrel, index = rationames, columns= ftypes)


print(count_frame)
print(count_frame_rel)
print(ratio_frame)
print(ratio_frame_reversed)
print(relative_ratio)

with open(os.path.join(res_fold, "all_extreme_ratios.txt"), "w") as f:
    print(ratio_frame.to_latex(),file=f)
with open(os.path.join(res_fold, "all_front_ratios.txt"), "w") as f:
    print(ratio_frame_reversed.to_latex(),file=f)
with open(os.path.join(res_fold, "front_counts.txt"), "w") as f:
    print(count_frame.to_latex(),file=f)
    print(file=f)
    print(count_frame_rel.to_latex(), file=f)
    