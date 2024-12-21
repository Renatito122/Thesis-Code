# 3D object detection for autonomous driving using a LiDAR based on active stereo with extended range
Thesis code used to build the model, the dictionaries and the use cases:

- DictionariesCreation
	- build1pImagesDictionary.py: Script to build a dictionary containg images of 1 projected point;
	- build2pImagesDictionary.py: Script to build a dictionary containg images of 2 projected points where, as the distance increases, the dots eventually merge, resulting in twice the number of photons; 

- Models
	- activeStereoModel.py: Script to simulate the laser's spot on the camera's sensor; 
	- proposedActiveStereoModel.py: Script to simulate two separated laser's spots on the camera's sensor;

- UC21
	- generateIdealStereoDatasetFirstEvenFrames.py: Script to build the new dataset related with UC2.1 (first 3750 even frames);
	- generateIdealStereoDatasetFirstOddFrames.py: Script to build the new dataset related with UC2.1 (first 3750 odd frames);
	- generateIdealStereoDatasetLastEvenFrames.py: Script to build the new dataset related with UC2.1 (last 3750 even frames);
	- generateIdealStereoDatasetLastOddFrames.py: Script to build the new dataset related with UC2.1 (last 3750 odd frames);
	- getImage.py: Script to generate Bird's-eye view of the first frame of the new dataset.

...

Use Cases:

- UC1: Ideal sensor;
- UC2.1: Ideal standard active stereo sensor;
- UC2.2: Standard active stereo sensor (with pixelization);
- UC2.3: Standard active stereo sensor (with pixelization and noise);
- UC3.1: Ideal proposed active stereo sensor;
- UC3.2: Proposed active stereo sensor (with pixelization);
- UC3.3: Proposed active stereo sensor (with pixelization and noise);
- UC4.1: Second version of proposed active stereo sensor (with pixelization);
- UC4.2: Second version of proposed active stereo sensor (with pixelization and noise).
