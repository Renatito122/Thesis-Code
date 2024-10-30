import numpy as np
import matplotlib.pyplot as plt


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
Npoints = int(2e4)

# Parameters for Lidar equation
ar = np.pi * (11.4e-3 / 2) ** 2
reflectivity = 10 / 100 # includes objet reflectivity and objective lens transmissivity

# Constants
h = 6.626e-34  # J*s
c = 299792458  # m/s

# Estimate the angular field of view of each pixel of the camera
Hfov = 2 * np.arctan(pxWidth * Horizontal_pixels / 2 / focal_length)
HfovPx = Hfov / Horizontal_pixels

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
		pixel_width = 2 * d * np.tan(HfovPx / 2) / 1.5
	else:
		pixel_width = 2 * d * np.tan(HfovPx / 2)


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

	# output_matrix = add_camera_noise(num_photons_per_pixel, noiseOn=True)

	return num_photons_per_pixel, xPx, yPx



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



def generate_calibration_matrix():

	# Intrinsic Camera Matrix
	K1 = np.array([[focal_length, 0, 0], [0, focal_length, 0], [0, 0, 1]])

	# Extrinsic Camera Matrices
	# Translation Matrix
	T1 = np.array([[0, 0, 0]]).T
	
	# Rotation Matrix
	R1 = np.identity(3)

	# Calibration Matrix
	P1 = np.matmul(K1, np.concatenate((R1, -T1), axis=1))

	return P1




if __name__ == "__main__":
	

	P1 = generate_calibration_matrix()

	# Read point cloud content
	with open(f'carPC.bin','rb') as pc:
		car_content = np.fromfile(pc, dtype=np.float32).reshape(-1,4)
		pc.close()

	with open(f'pedPC.bin','rb') as pc:
		ped_content = np.fromfile(pc, dtype=np.float32).reshape(-1,4)
		pc.close()


	l_image_point = []
	color = []

	for index, point in enumerate(car_content):

		if np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2]):
			continue

		if point[3] == 1 and ped_content[index][3] == 0: 
			color.append('#ff0000')
		elif point[3] == 0 and ped_content[index][3] == 1:
			color.append('#0000ff')
		else:
			color.append('#000000')

		# Rearrange point to fit to camera
		point = np.array([point[1], point[2], point[0]]).T
		point = np.hstack((point, np.ones((1,))))

		# Camera projetion coordinates
		x1 = np.matmul(P1, point)

		# Normalization
		if x1[2] != 0:
			x1 /= x1[2]

		l_image_point.append(x1)


	l_image_point = np.array(l_image_point)
 
 
	fig = plt.figure(figsize=(6*2,4*1))
	plt.subplot(1,2,1)
	plt.scatter(l_image_point[:, 0],l_image_point[:,1], s=0.025, c = color)
	plt.title('Ideal camera image')
	plt.xlabel('x [m]')
	plt.ylabel('y [m]')

	plt.subplot(1,2,2)
	plt.scatter(l_image_point[:, 0]/pxWidth,l_image_point[:,1]/pxWidth, s=0.025, c = color)
	plt.title('Ideal camera image')
	plt.xlabel('x [px]')
	plt.ylabel('y [px]')

	plt.tight_layout()
	plt.show()
 
 
	# Generate a grid of pixels ready to store a gray scale image
	xPxGrid = np.arange(np.ceil(np.min(l_image_point[:, 0]/pxWidth))-5,np.ceil(np.max(l_image_point[:, 0]/pxWidth))+5)
	yPxGrid = np.arange(np.ceil(np.min(l_image_point[:, 1]/pxWidth))-5,np.ceil(np.max(l_image_point[:, 1]/pxWidth))+5)
	XPx,YPx = np.meshgrid(xPxGrid,yPxGrid)
	NphotonsMatrix = np.zeros((len(yPxGrid),len(xPxGrid)))
 
	# Get real world points
	realWorldPoint  = np.array([car_content[:,1], car_content[:,2], car_content[:,0]])

	# Remove NaNs
	realWorldPoint = realWorldPoint[:,~np.isnan(realWorldPoint).any(axis=0)]

	for a in range(len(l_image_point)):
     
		# Load points
		imagePlanePoint = l_image_point[a]
  
		# Calculate distance
		r = np.round(np.sqrt((realWorldPoint[0][a] ** 2) + (realWorldPoint[1][a] ** 2) + (realWorldPoint[2][a] ** 2)))

		if r % 2 != 0:
			r += 1

		# Remove the integer part of the x coordinate
		cx_px = imagePlanePoint[0] / pxWidth

		# cx_shift = np.round(cx_px)

		# cx_decimal = np.round(cx_px - cx_shift, 1)

		# if cx_decimal > 0.5:
		# 	cx_decimal -= 1
		# 	cx_shift -= 1							
		# elif cx_decimal < -0.5:
		# 	cx_decimal += 1
		# 	cx_shift += 1


		# Remove the integer part of the y coordinate
		cy_px = imagePlanePoint[1] / pxWidth

		# cy_shift = np.round(cy_px)

		# cy_decimal = np.round(cy_px - cy_shift, 1)

		# if cy_decimal > 0.5:
		# 	cy_decimal -= 1
		# 	cy_shift -= 1							
		# elif cy_decimal < -0.5:
		# 	cy_decimal += 1
		# 	cy_shift += 1

		# Estimate the number of photons per pixel
		NphotonsForEachPixel, xPx, yPx = pixelization(d = r, aux_x = cx_px, aux_y = cy_px)

    	# Add such a number of photons to the image
		NphotonsMatrix[int(yPx[0] - np.min(yPxGrid)):int(yPx[-1] - np.min(yPxGrid)+ 1),
				 int(xPx[0] - np.min(xPxGrid)):int(xPx[-1] - np.min(xPxGrid )+ 1)] += NphotonsForEachPixel
     


	# Plot image without noise
	plt.subplots()
	# plt.imshow(NphotonsMatrix[::-1,:], cmap='gray')
	plt.imshow(NphotonsMatrix[-170:-210:-1,3050:3100], cmap='gray')
	plt.title('Number of photons per px')
	plt.xlabel('x [px]')
	plt.ylabel('y [px]')
	cbar = plt.colorbar()
	# plt.xlim(3000,3150)
	# plt.ylim(300,100)
	plt.show()

	
	# Apply noise function
	imgOut = add_camera_noise(NphotonsMatrix)


	# Plot small piece of the noisy image
	fig = plt.figure(figsize=(6*2,4*3))
	plt.subplot(2,1,1)
	plt.imshow(imgOut[-170:-210:-1,3050:3100], cmap='jet')
	plt.title('Image with noise')
	plt.xlabel('x [px]')
	plt.ylabel('y [px]')
	cbar = plt.colorbar()

	plt.subplot(2,1,2)
	plt.imshow(imgOut[-170:-210:-1,3050:3100] > 200, cmap='jet')
	plt.title('Pixels with more than 200 ADUs')
	plt.xlabel('x [px]')
	plt.ylabel('y [px]')
	cbar = plt.colorbar()

	plt.tight_layout()
	plt.show()


	# Plot full noisy image
	fig = plt.figure(figsize=(6*2,4*3))
	plt.subplot(2,1,1)
	plt.imshow(imgOut[::-1,:], cmap='jet')
	plt.title('Image with noise')
	plt.xlabel('x [px]')
	plt.ylabel('y [px]')
	cbar = plt.colorbar()

	plt.subplot(2,1,2)
	plt.imshow(imgOut[::-1,:] > 200, cmap='jet')
	plt.title('Pixels with more than 200 ADUs')
	plt.xlabel('x [px]')
	plt.ylabel('y [px]')
	cbar = plt.colorbar()
	

	plt.tight_layout()
	plt.show()