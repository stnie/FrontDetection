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

# Usage of the provided network
The trained network can be tested on NWS data using the provided scripts. All scripts assume that the necessary ERA5 data is located directly in the corresponding ERA5_Data folder, while labels are located at Label_Data/hires/. Corresponding input and label files are assumend to have the same name except for the extension. Label data should be provided as ".txt" file in a format according to the High Resolution Coded Surface Bulletins issued by the NWS. 

Potential command line command, assuming you are currently located in the Scripts_and_Examples Folder:
Create_Climatology.sh path/to/network/<network_name>.pth  /path/to/network/data_set_info.txt <output_name>

This will create a subfolder <output_name> in the folder according to the used script (Climatologies in this case), which contains several output files. ".bin" files are binary dumps of float32 data. In the case of Climatologies they have a resolution of 720x1440. In the case of Cross Sections they have a resolution of 17x4.  
