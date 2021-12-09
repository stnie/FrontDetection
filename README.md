# FrontDetection
CNN for detection and classification of synoptic scale fronts from ERA5 Reanalysis data
The provided Code was tested on Python 3.8.2 and Pytorch 1.6

The provided network detects and classifies synoptic scale fronts from atmospheric variables into 4 classes (warm front, cold front, occlusion, stationary front)

# Input Data
The network runs on multilevel - multivariable input from ERA5 Reanalysis data on a regular grid. 
The network was trained on levels 105,109,113,117,121,125,129,133,137 according to the L137 level definition.
Used variables are t,q,u,v,w,sp and size of a pixel in longitudinal direction. The latter is derived from the variable latitude.  

# Output
The provided output has the same resolution as the input data, where for each pixel a raw score for each of the 5 output channels (background, warm, cold, occlusion, stationary) is provided. To generate probabilities for each output channel the application of a softmax operation is recommended.

# Trained example
The pretrained example was trained using input samples from 2012 to 2015 and 2018 to 2019 against labeled data generated from the Deutscher Wetterdienst (DWD) and North American Weather Service (NWS). The Network was validated on data from 2017 (not used for training) and tested on data from 2016 (not used for training)

# Scores and Output examples
The trained network generates a Critical Success Index against the weather service labels of more than 60\% for the detection of fronts. An exemplary timelapse of the networks output can be seen at https://av.tib.eu/media/54716 for January 2016 at a 1 hour resolution. 


# Installation
Clone the repository for access to the code. For access to the pretrained models use git lfs to additionally pull the pretrained models. 
For evaluation purposes no further installation is necessary apart from preparing your datasets. If you want to train a network or use the loss as described in the paper, you need to run the RunCModulesSetup.sh script to compile the matching algorithm used in the proposed loss function. PyBind11 is needed for this. The provided Scripts use fixed output locations, which need to be Created first. 
Navigate into the Scripts and Examples Subfolder and Create 4 Folders:  Predictions, Climatologies, CrossSections and OutputImages

e.g.
git clone <this_repository>

git lfs pull 

cd Scripts_and_Examples

./RunCModulesSetup.sh

mkdir Predictions

mkdir Climatologies

mkdir CrossSections

mkdir OutputImages

# Usage of the provided network
The trained network can be tested on NWS data using the provided scripts.  

All scripts assume that the necessary ERA5 data is located at <path/to/DataFolder> without any subfolders, while labels are located at <path/to/LabelFolder>/hires/. 

Corresponding input and label files are assumend to have the same name except for the extension. Label data should be provided as ".txt" file in a format according to the High Resolution Coded Surface Bulletins issued by the NWS. 

Potential command line command, assuming you are currently located in the Scripts_and_Examples Folder:

Create_Output_Samples_raw.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name> <path/to/DataFolder> <path/to/LabelFolder>

Create_Climatology.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name> <path/to/DataFolder> <path/to/LabelFolder>

Calculate_CSI.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name> <path/to/DataFolder> <path/to/LabelFolder> [--preCalc]

Create_Cross_Section.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name> <variable> <path/to/DataFolder> <path/to/LabelFolder> <path/to/variableFolder> [--preCalc]

Create_Output_Samples.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name> <path/to/DataFolder> <path/to/LabelFolder> [--preCalc]

<variable> is a string corresponding to the variable that should be extracted from the ERA5 file, with a corresponding suffix to detail which type of ERA5 File should be used. t_ml for example extracts t from a file named mlYYYYMMDD_HH.nc, while q_z extracts q from a file named ZYYYYMMDD_HH.nc. Adding a "d" as a prefix uses the derivative instead (e.g. dt or dq). Potential suffixes are : _ml , _z and _b. Note that at this point only a very restricted selection of variables, files and height levels is directly supported (e.g. t, q, rq, ept, wind, tfp; with most of these being available for the _b suffix). This will be adjusted in a future release.

<variableFolder> is a folder containing ERA5 data, where the corresponding <variable> should be extracted from. 
The data is assumed to be organized in multiple folders <variableFolder/YYYY/MM/dataname>. Note that these <dataname> is assumend to be prefixed by the suffix of <variable> (e.g. <variableFolder>/2016/02/ml20160201_00.nc is the file to extract t_ml at 1st February 2016 , 00 UTC)

For best results we propose to use larger input dimensions, to reduce the effect of critical information being cropped.

the option [--preCalc] can be used, when the result (folder) of "Create_Output_Samples_raw.sh" is used as data input, to omit GPU-Inference. The results are then directly read from disk instead. Note: Create_Output_Samples_raw.sh creates fullsize (720 x 1440 x 5) outputs to minimize cropping issues. Reading those pre calculated results also assumes that the files have these dimensions! If the first output channel shall be filtered to only contain information from a subset of the four classification channels (e.g. to create a climatology omitting stationary fronts) [--preCalc] should not be used! The subset filtering adjusts the predicted probabilities. The pre-Calculated results however are saved after thresholding, thus no longer containing the exact probability information.

This will create a subfolder <output_name> in the folder according to the used script, which contains several output files. 
".bin" files are binary dumps of float32 data. 
In the case of Climatologies they have a resolution of 720x1440. 
In the case of Cross Sections they have a resolution of 21x4.  


# Training of a network
Training of a network can be performed using "train.sh" script.
data should be split into training and validation sets and be located in the Training_Data/data  or Training_Data/test_data folder. Label data should be split identically and be located in the Training_Data/label or Training_Data/test_label folder. Naming and format of Label according to the previous section.
