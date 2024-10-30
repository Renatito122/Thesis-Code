import numpy as np
import matplotlib.pyplot as plt
import pickle


# Laser parameters
laser_diameter = 4.5e-3
laser_divergence = 0.2e-3
wavelength = 850e-9

# Camera parameters
focal_length = 16e-3  # in meters
pxWidth = 5.86e-6
Horizontal_pixels = 1936
camera_width = 11.345e-3
camera_height = 7.126e-3

# Camera FOV
camera_diagonal = np.sqrt(camera_width**2 + camera_height**2)
camera_FOV = 2 * np.degrees(np.arctan(camera_diagonal / (2 * focal_length)))

# Pulse Energy calculation
laser_power = 100e-6
frame_rate = 50
pulse_energy = laser_power / frame_rate

# Number of dots
Npoints = int(1e5)

# Parameters for Lidar equation
ar = np.pi * (11.4e-3 / 2) ** 2
reflectivity = 10 / 100  # includes object reflectivity and objective lens transmissivity

# Constants
h = 6.626e-34  # J*s
c = 299792458  # m/s

# Field of view in radians per pixel
AfovPxRad = np.arctan(pxWidth/focal_length) 			# in degrees

# Parameters for noise model
quantum_efficiency = 6.25 / 100
sensitivity = 1.923
dark_noise = 6.83
bit_depth = 12
baseline = 100
maximum_adu = int(2 ** bit_depth - 1)
seed = 42
rs = np.random.RandomState(seed)

# Distance between lasers
lasers_distance = 11.7e-3

# Calculate distance where overlap starts
points_distance = lasers_distance - laser_diameter		# without divergence
fov_per_px = (camera_FOV / Horizontal_pixels)*(np.pi/180)
overlap_distance = int(points_distance / (np.tan(fov_per_px)))
# print(overlap_distance)




# Noise model function
def add_camera_noise(num_photons, noiseOn=True):

    if noiseOn:

        # Add photon shot noise
        photons = rs.poisson(num_photons, size=num_photons.shape)

        # Get the number of photoelectrons
        num_photoelectrons = np.round(quantum_efficiency * photons)

        # Add dark noise
        electrons_out = np.round(rs.normal(scale=dark_noise, size=num_photoelectrons.shape) + num_photoelectrons)

        # Convert electrons to Analog-to-Digital Units (ADU) and add baseline
        adu = (electrons_out * sensitivity).astype(int)  # Ensure the final ADU count is discrete
        adu += baseline
        adu[adu > maximum_adu] = maximum_adu  # Models pixel saturation

    else:

        # Convert electrons to Analog-to-Digital Units (ADU)
        adu = (num_photons * quantum_efficiency * sensitivity).astype(int)
        adu[adu > maximum_adu] = maximum_adu  # Models pixel saturation

    return adu



def pixelization(d, side, aux_x, aux_y):

    # Estimate laser dot diameter
    spot_diameter = laser_diameter + 2 * d * np.tan(laser_divergence / 2)

    # Estimate pixel width
    pixel_width = 2 * d * np.tan(AfovPxRad / 2)

    # Total number of received photons
    free_space_path_loss = 1 / (4 * np.pi * d ** 2)
    energy_captured_by_camera = pulse_energy * free_space_path_loss * ar * reflectivity
    photon_energy = h * c / wavelength
    num_photons = energy_captured_by_camera / photon_energy

    # Generate circle based on point
    # The "side" parameter is used to determine if the laser circumference will be generated on the left (negative x offset) or right side (positive x offset)

    # Generate the laser circumference
    if side == 0:
        # Left Side (Circle CenterX = -laser_separation/2)
        x = np.linspace(-(spot_diameter / 2), +(spot_diameter / 2), 256)	# 256 points
        y = np.sqrt((spot_diameter**2 / 4) - x**2)
        x = x - lasers_distance/2
    else:
        # Right Side (Circle CenterX = laser_separation/2)
        x = np.linspace(-(spot_diameter / 2), +(spot_diameter / 2), 256)	# 256 points
        y = np.sqrt((spot_diameter**2 / 4) - x**2)
        x = x + lasers_distance/2

    x = x + 0
    y = y + 0

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


    return num_photons_per_pixel, xPx, yPx




if __name__ == "__main__":
      

	#############################################################################################
	######################### CODE FOR IMAGES ###################################################
	#############################################################################################

      
    
	# for d in range(2,71,2):
	# 	print('Distance = ' + str(d) + ' m')
	# 	point = [0, 0, d, 1]

	# 	# Get the images for left and right spots
	# 	num_photons_per_pixel_left, x_left, y_left = pixelization(d, 0, 0, 0)
	# 	num_photons_per_pixel_right, x_right, y_right = pixelization(d, 1, 0, 0)
  

	# 	# Normalize x and y coordinates (to avoid negative indexes)
	# 	x_leftNorm = x_left + abs(min(x_left))
	# 	x_rightNorm = x_right + abs(min(x_left))
	# 	y_leftNorm = y_left + abs(min(y_left))
	# 	y_rightNorm = y_right + abs(min(y_right))

	# 	# Create the final image
	# 	Xfinal, Yfinal = np.meshgrid(np.arange(min(x_leftNorm), max(x_rightNorm)+1), np.arange(y_leftNorm[0], y_leftNorm[-1]+1))
	# 	finalMatrixNumberOfPhotons = np.zeros_like(Xfinal)

	# 	# Convert to numpy array
	# 	x_leftNorm = np.array(x_leftNorm)
	# 	x_rightNorm = np.array(x_rightNorm)

	# 	# Left Spot
	# 	for x in range(int(x_leftNorm[0]), int(x_leftNorm[-1]+1)):
	# 		for y in range(int(y_leftNorm[0]), int(y_leftNorm[-1]+1)):
	# 			finalMatrixNumberOfPhotons[y, x] += num_photons_per_pixel_left[y, x]

	# 	# Right Spot
	# 	for x in range(int(x_rightNorm[0]), int(x_rightNorm[-1]+1)):
	# 		for y in range(int(y_rightNorm[0]), int(y_rightNorm[-1]+1)):
	# 			finalMatrixNumberOfPhotons[y, x] += num_photons_per_pixel_right[y, x - int(x_rightNorm[0])]

	# 	# Add noise to the final image
	# 	#noisyImage = add_camera_noise(finalMatrixNumberOfPhotons, noiseOn=True)

	# 	# Add noise to the left and right spots (Purely for ADU comparison)
	# 	#leftSpot = add_camera_noise(num_photons_per_pixel_left, noiseOn=True)
	# 	#rightSpot = add_camera_noise(num_photons_per_pixel_right, noiseOn=True)
	# 	#print('Max ADU: ' + str(max(noisyImage.flatten())))
	# 	#print('Left Max ADU: ' + str(max(leftSpot.flatten())))
	# 	#print('Right Max ADU: ' + str(max(rightSpot.flatten())))

	# 	fig, ax = plt.subplots()
	# 	img = ax.imshow(finalMatrixNumberOfPhotons, interpolation='nearest')

	# 	ax.set_xticks(np.arange(min(x_leftNorm) - 2.5, max(x_rightNorm) + 3.5, 3))
	# 	ax.set_yticks(np.arange(min(y_leftNorm) - 2.5, max(y_leftNorm) + 3.5, 3))  

	# 	ax.set_xticklabels(np.arange(min(x_leftNorm) - 3, max(x_rightNorm) + 3, 3))
	# 	ax.set_yticklabels(np.arange(min(y_leftNorm) - 3, max(y_leftNorm) + 3, 3))

	# 	ax.set_xticks(np.arange(min(x_leftNorm) - 2.5, max(x_rightNorm) + 3.5, 1))
	# 	ax.set_yticks(np.arange(min(y_leftNorm) - 2.5, max(y_leftNorm) + 3.5, 1)) 
			

	# 	# Annotate each cell with its value
	# 	for i in range(finalMatrixNumberOfPhotons.shape[0]):
	# 		for j in range(finalMatrixNumberOfPhotons.shape[1]):
	# 			ax.text(j, i, int(finalMatrixNumberOfPhotons[i, j]), ha='center', va='center', color='black')


	# 	cb = plt.colorbar(img)
	# 	cb.set_label('ADU')

	# 	# Enable grid
	# 	ax.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
		
	# 	ax.set_xlim(x_leftNorm[0] - 2.5, x_rightNorm[-1] + 2.5)
	# 	ax.set_ylim(x_leftNorm[0] - 2.5, x_leftNorm[-1] + 2.5)
	# 	plt.show()
            

	##############################################################################################
	##############################################################################################
	##############################################################################################	
      

	#############################################################################################
	######################### CODE FOR DICTIONARIES BUILD #######################################
	#############################################################################################


	# Create dictionary for images without noise
	no_noise_images = {}

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
					
					# Get the images for left and right spots
					num_photons_per_pixel_left, x_left, y_left = pixelization(d, 0, aux_x, aux_y)
					num_photons_per_pixel_right, x_right, y_right = pixelization(d, 1, aux_x, aux_y)

					# Normalize x and y coordinates (to avoid negative indexes)
					x_leftNorm = x_left + abs(min(x_left))
					x_rightNorm = x_right + abs(min(x_left))
					y_leftNorm = y_left + abs(min(y_left))
					y_rightNorm = y_right + abs(min(y_right))

					# Create the final image
					Xfinal, Yfinal = np.meshgrid(np.arange(min(x_leftNorm), max(x_rightNorm)+1), np.arange(y_leftNorm[0], y_leftNorm[-1]+1))
					finalMatrixNumberOfPhotons = np.zeros_like(Xfinal)

					# Convert to numpy array
					x_leftNorm = np.array(x_leftNorm)
					x_rightNorm = np.array(x_rightNorm)

					# Left Spot
					for x in range(int(x_leftNorm[0]), int(x_leftNorm[-1]+1)):
						for y in range(int(y_leftNorm[0]), int(y_leftNorm[-1]+1)):
							finalMatrixNumberOfPhotons[y, x] += num_photons_per_pixel_left[y, x]

					# Right Spot
					for x in range(int(x_rightNorm[0]), int(x_rightNorm[-1]+1)):
						for y in range(int(y_rightNorm[0]), int(y_rightNorm[-1]+1)):
							finalMatrixNumberOfPhotons[y, x] += num_photons_per_pixel_right[y, x - int(x_rightNorm[0])]

					no_noise_images[aux_x][aux_y][d] = np.array(finalMatrixNumberOfPhotons)

	# Write dictionary to file
	with open("2point_no_noise_images.pkl", "wb") as f:
		pickle.dump(no_noise_images, f)
		f.close()
		

	##############################################################################################
	##############################################################################################
	##############################################################################################	