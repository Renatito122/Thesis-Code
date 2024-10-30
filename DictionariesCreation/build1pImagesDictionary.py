import numpy as np
import matplotlib.pyplot as plt
import pickle

# Laser parameters
laser_diameter = 4.5e-3
laser_divergence = 0.2e-3
wavelength = 850e-9

# Camera parameters
focal_length = 16e-3 # in meters
pxWidth = 5.86e-6
Horizontal_pixels = 1936

# Pulse Energy calculation
laser_power = 100e-6
frame_rate = 50
pulse_energy = laser_power / frame_rate

# Number of dots
Npoints = int(1e5)

# Parameters for Lidar equation
ar = np.pi * (11.4e-3 / 2) ** 2
reflectivity = 10 / 100 # includes objet reflectivity and objective lens transmissivity

# Constants
h = 6.626e-34  # J*s
c = 299792458  # m/s

# Field of view in radians per pixel
AfovPxRad = np.arctan(pxWidth/focal_length) 			# in degrees

# Parameters for noise model
quantum_efficiency = 6.25/100
sensitivity = 1.923
dark_noise = 6.83
bit_depth = 12
baseline = 100
maximum_adu = int(2**bit_depth - 1)
seed = 42
rs = np.random.RandomState(seed)



def pixelization(d, aux_x, aux_y):

	# Estimate laser dot diameter
	spot_diameter = laser_diameter + 2 * d * np.tan(laser_divergence / 2)

	# Estimate pixel width
	if 4 <= d <= 16:
		pixel_width = 2 * d * np.tan(AfovPxRad / 2) / 1.5
	else:
		pixel_width = 2 * d * np.tan(AfovPxRad / 2)


	# Total number of received photons
	free_space_path_loss = 1 / (4 * np.pi * d**2)
	energy_captured_by_camera = pulse_energy * free_space_path_loss * ar * reflectivity
	photon_energy = h * c / wavelength
	num_photons = energy_captured_by_camera / photon_energy

	#print(num_photons)


	# Generate circle based on point
	# Generate the laser circumference
	x = np.linspace(-spot_diameter / 2, + spot_diameter / 2, 256)
	y = np.sqrt(spot_diameter**2 / 4 - x**2)

	# Generate laser dot
	x = np.hstack((x, x[::-1]))
	y = np.hstack((y, -y))

	# Identify which pixels are illuminated by the laser dots
	x1minPx = np.floor(np.nanmin(x / pixel_width) + aux_x)
	x1maxPx = np.ceil(np.nanmax(x / pixel_width) + aux_x)
	y1minPx = np.floor(np.nanmin(y / pixel_width) + aux_y)
	y1maxPx = np.ceil(np.nanmax(y / pixel_width) + aux_y)

	# Identify the meshgrid of pixels
	xPx = np.arange(x1minPx, x1maxPx + 1)
	yPx = np.arange(y1minPx, y1maxPx + 1)
	XPx, YPx = np.meshgrid(xPx, yPx)

	# Fill the dot with dots
	circleRadius = spot_diameter / pixel_width / 2
	circleDots = (np.random.rand(Npoints) * circleRadius * np.exp(1j * 2 * np.pi * np.random.rand(Npoints)))
	circleDotsx = np.real(circleDots) + np.mean(x) / pixel_width
	circleDotsy = np.imag(circleDots)

	# Generate laser dot
	circleDotsx += aux_x
	circleDotsy += aux_y

	# Shift x and y
	x += aux_x * pixel_width
	y += aux_y * pixel_width

	# Calculate number of photons in each pixel
	# Sweep each pixel and estimate the number of photons that fall on it
	dotsInPixel = np.zeros((len(yPx), len(xPx), Npoints), dtype=bool)

	for a in range(len(yPx)):
		for b in range(len(xPx)):
			pixelLimits = [XPx[a, b] - 0.5, XPx[a, b] + 0.5, YPx[a, b] - 0.5, YPx[a, b] + 0.5]

			# Identify the dots that fall on the pixel
			dotsInPixel[a, b, :] = np.logical_and(np.logical_and(circleDotsx >= pixelLimits[0], circleDotsx <= pixelLimits[1]),
										 np.logical_and(circleDotsy >= pixelLimits[2], circleDotsy <= pixelLimits[3]))

	# Count (and normalize) the number of dots in each pixel
	dotsInPixel = np.sum(dotsInPixel, axis=2) / Npoints

	# Pixelization
	# Total number of photons in each pixel
	num_photons_per_pixel = num_photons * dotsInPixel

	#print(num_photons_per_pixel)

	return num_photons_per_pixel



# Noise model function
def add_camera_noise(num_photons, noiseOn=True):

	if noiseOn:

		# Add photon shot noise
		photons = rs.poisson(num_photons, size = num_photons.shape)
		
		# Get the number of photoelectrons
		num_photoelectrons = np.round(quantum_efficiency * photons)
		
		# Add dark noise
		electrons_out = np.round(rs.normal(scale = dark_noise, size = num_photoelectrons.shape) + num_photoelectrons)
		
		# Convert electrons to Analog-to-Digital Units (ADU) and add baseline
		adu = (electrons_out * sensitivity).astype(int) # Ensure the final ADU count is discrete
		adu += baseline
		adu[adu > maximum_adu] = maximum_adu # Models pixel saturation
	
	else:

		# Convert electrons to Analog-to-Digital Units (ADU)
		adu = (num_photons * quantum_efficiency * sensitivity).astype(int)
		adu[adu > maximum_adu] = maximum_adu # Models pixel saturation
	
	return adu



if __name__ == "__main__":


	#############################################################################################
	######################### CODE FOR IMAGES ###################################################
	#############################################################################################


	# vmin = None
	# vmax = None

	# for d in range(2, 71, 2):
	# 	print('distance = ' + str(d) + ' m')
	# 	point = [0, 0, d, 1]

	# 	imageOut = pixelization(d, 0, 0)

	# 	print(imageOut)
	# 	print(type(imageOut))
    
	# 	noisyImageOut = add_camera_noise(imageOut, noiseOn=True) 

	# 	# print(noisyImageOut)

	# 	if vmin is None or np.min(noisyImageOut) < vmin:
	# 		vmin = np.min(noisyImageOut)
	# 	if vmax is None or np.max(noisyImageOut) > vmax:
	# 		vmax = np.max(noisyImageOut)

	# 	fig, ax = plt.subplots()
	# 	img = ax.imshow(noisyImageOut, vmin=vmin, vmax=vmax)
	# 	ax.set_xticks([])
	# 	ax.set_yticks([])

	# 	# Annotate each cell with its value
	# 	for i in range(noisyImageOut.shape[0]):
	# 		for j in range(noisyImageOut.shape[1]):
	# 			ax.text(j, i, str(noisyImageOut[i, j]), ha='center', va='center', color='black')

	# 	cb = plt.colorbar(img)
	# 	cb.set_label('ADU')

	# 	plt.show()


	##############################################################################################
	##############################################################################################
	##############################################################################################					



	#############################################################################################
	######################### CODE FOR DICTIONARIES BUILD #######################################
	#############################################################################################


	# Create dictionary for images without noise
	no_noise_images = {}

	# Iterate over all distances (integer) between 2 and 70 meters
	for d in range(2, 71, 2):
		print('distance = ' + str(d) + ' m')
		point = [0, 0, d, 1]

		# Iterate over every one decimal case combination of zero for x
		for aux_x in [n / 10.0 for n in range(-5, 6)]:
			if aux_x not in no_noise_images:
				no_noise_images[aux_x] = {}


			# Iterate over every one decimal case combination of zero for y
			for aux_y in [m / 10.0 for m in range(-5, 6)]:
				if aux_y not in no_noise_images[aux_x]:
					no_noise_images[aux_x][aux_y] = {}

				# Perform 1 realization
				for a in range(1):

					noNoiseImage = pixelization(d, aux_x, aux_y)

					no_noise_images[aux_x][aux_y][d] = np.array(noNoiseImage)

	# Write dictionary to file
	with open("no_noise_images.pkl", "wb") as f:
		pickle.dump(no_noise_images, f)
		f.close()



	# # Create dictionary for images with noise
	# noise_images = {}

	# # Iterate over all distances (integer) between 2 and 70 meters
	# for d in range(2, 71, 2):
	# 	print('distance = ' + str(d) + ' m')
	# 	point = [0, 0, d, 1]

	# 	# Iterate over every one decimal case combination of zero for x
	# 	for aux_x in [n / 10.0 for n in range(-5, 6)]:
	# 		if aux_x not in noise_images:
	# 			noise_images[aux_x] = {}


	# 	# Iterate over every one decimal case combination of zero for y
	# 		for aux_y in [m / 10.0 for m in range(-5, 6)]:
	# 			if aux_y not in noise_images[aux_x]:
	# 				noise_images[aux_x][aux_y] = {}

	# 			# Perform 10 realizations
	# 			for a in range(10):

	# 				noNoiseImage, XPx, YPx = pixelization(d, aux_x, aux_y)
	# 				noisyImage = add_camera_noise(noNoiseImage, noiseOn=True)

	# 				if d in noise_images[aux_x][aux_y]:
	# 					noise_images[aux_x][aux_y][d].append(np.array(noisyImage))

	# 				else:
	# 					noise_images[aux_x][aux_y][d] = [np.array(noisyImage)]

	# # Write dictionary to file
	# with open("noise_images.pkl", "wb") as f:
	# 	pickle.dump(noise_images, f)
	# 	f.close()




	# # Create dictionary for noise centroids
	# # Calculate threshold
	# image = np.zeros((100,100))

	# noisyimage = add_camera_noise(image, noiseOn=True)

	# # Flatten the array
	# flattened_data = noisyimage.flatten()

	# # Create the histogram
	# plt.hist(flattened_data, bins=40, color='blue', alpha=0.7)
	# plt.title('Noise interval')
	# plt.xlabel('Values')
	# plt.ylabel('Frequency')
	# plt.grid(True)
	# plt.show()


	# mean = np.mean(flattened_data)
	# std = np.std(flattened_data)

	# threshold = mean + 2*std

	# print(threshold)

	# # Create the histogram
	# plt.hist(flattened_data, bins=40, color='blue', alpha=0.7)
	# plt.title('Histogram of Data')
	# plt.xlabel('Values')
	# plt.ylabel('Frequency')
	# plt.grid(True)
	# plt.axvline(x=threshold, color='red', linestyle='--', label='Mean + 2*std')
	# plt.legend()
	# plt.show()


	# noise_centroids = {}

	# # Iterate over all distances (integer) between 2 and 70 meters
	# for d in range(2, 71, 2):
	# 	print('distance = ' + str(d) + ' m')
	# 	point = [0, 0, d, 1]

	# 	# Iterate over every one decimal case combination of zero for x
	# 	for aux_x in [np.round(n / 10.0, 1) for n in range(-5, 6)]:
	# 		if aux_x not in noise_centroids:
	# 			noise_centroids[aux_x] = {}


	# 		# Iterate over every one decimal case combination of zero for y
	# 		for aux_y in [np.round(m / 10.0, 1) for m in range(-5, 6)]:
	# 			if aux_y not in noise_centroids[aux_x]:
	# 				noise_centroids[aux_x][aux_y] = {}

	# 			# Perform 10 realizations
	# 			for a in range(10):

	# 				noNoiseImage, XPx, YPx = pixelization(d, aux_x, aux_y)
	# 				noisyImage = add_camera_noise(noNoiseImage, noiseOn=True)

	# 				# Create a boolean mask where True represents values higher than the threshold
	# 				mask = noisyImage > threshold

	# 				# Count the number of True values in the mask
	# 				count = np.sum(mask)

	# 				if count == 0:
	# 					continue

	# 				else:

	# 					# Calculate centroid
	# 					centroid_x = np.sum(XPx * noisyImage) / np.sum(noisyImage)
	# 					centroid_y = np.sum(YPx * noisyImage) / np.sum(noisyImage)

	# 					if d in noise_centroids[aux_x][aux_y]:
	# 						noise_centroids[aux_x][aux_y][d].append([centroid_x, centroid_y])

	# 					else:
	# 						noise_centroids[aux_x][aux_y][d] = [[centroid_x, centroid_y]]

	# # Write dictionary to file
	# with open("noise_centroids.pkl", "wb") as f:
	# 	pickle.dump(noise_centroids, f)
	# 	f.close()



    ##############################################################################################
	##############################################################################################
	##############################################################################################