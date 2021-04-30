# FrontDetection
CNN for detection and classification of synoptic scale fronts from ERA5 Reanalysis data
The provided Code was tested on Python 3.8.2 and Pytorch 1.6

The provided network detects and classifies synoptic scale fronts from atmospheric variables into 4 classes (warm front, cold front, occlusion, stationary front)

# Input Data
The network runs on multilevel - multivariable input from ERA5 Reanalysis data on a regular grid. 
The network was trained on levels 105,109,113,117,121,125,129,133,137 according to the L137 level definition.
Used variables are t,q,u,v,w,sp and size of a pixel in longitudinal direction.

# Output
The provided output has the same resolution as the input data, where for each pixel a raw score for each of the 5 output channels (background, warm, cold, occlusion, stationary) is provided. To generate probabilities for each output channel the application of a softmax operation is recommended.

# Trained example
The pretrained example was trained using input samples from 2012 to 2015 and 2018 to 2019 against labeled data generated from the Deutscher Wetterdienst (DWD) and North American Weather Service (NWS). The Network was validated on data from 2017 (not used for training) and tested on data from 2016 (not used for training)

# Scores and Output examples
The trained network generates a Critical Success Index against the weather service labels of more than 60\% for the detection of fronts. An exemplary timelapse of the networks output can be seen at ... for January 2016 at a 1 hour resolution. 


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
The trained network can be tested on NWS data using the provided scripts. All scripts assume that the necessary ERA5 data is located at <path/to/DataFolder> without any subfolders, while labels are located at <path/to/LabelFolder>/hires/. Corresponding input and label files are assumend to have the same name except for the extension. Label data should be provided as ".txt" file in a format according to the High Resolution Coded Surface Bulletins issued by the NWS. 

Potential command line command, assuming you are currently located in the Scripts_and_Examples Folder:
Create_Climatology.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name> <path/to/DataFolder> <path/to/LabelFolder>
Calculate_CSI.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name> <path/to/DataFolder> <path/to/LabelFolder>
Create_Cross_Section.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name> <variable> <path/to/DataFolder> <path/to/LabelFolder>
Create_Output_Samples.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name> <path/to/DataFolder> <path/to/LabelFolder>

This will create a subfolder <output_name> in the folder according to the used script, which contains several output files. 
".bin" files are binary dumps of float32 data. 
In the case of Climatologies they have a resolution of 720x1440. 
In the case of Cross Sections they have a resolution of 17x4.  

# Training of a network
Training of a network can be performed using "train.sh" script.
data should be split into training and validation sets and be located in the Training_Data/data  or Training_Data/test_data folder. Label data should be split identically and be located in the Training_Data/label or Training_Data/test_label folder. Naming and format of Label according to the previous section.
