import numpy as np
import cv2 as openCV
import pickle
import matplotlib.pyplot as plt


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
reflectivity = 10 / 100 # includes objet reflectivity and objective lens transmissivity

# Constants
h = 6.626e-34  # J*s
c = 299792458  # m/s

# Field of view in radians per pixel
AfovPxRad = np.arctan(pxWidth/focal_length)

# Parameters for noise model
quantum_efficiency = 6.25/100
sensitivity = 1.923
dark_noise = 6.83
bit_depth = 12
baseline = 100
maximum_adu = int(2**bit_depth - 1)
seed = 42
rs = np.random.RandomState(seed)

# Distance between the 2 cameras
baselineDistance = 1.2

# Translation vector between two dots of each set of dots
dotTranslation = [11.7e-3, 0, 0, 0]

# Calculate distance where overlap starts
points_distance = dotTranslation[0] - laser_diameter		# without divergence
fov_per_px = (camera_FOV / Horizontal_pixels)*(np.pi/180)
overlap_distance = int(points_distance / (np.tan(fov_per_px)))
# print(overlap_distance)


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
	K2 = K1.copy()

	# Extrinsic Camera Matrices
	# Translation Matrix
	T1 = np.array([[-baselineDistance/2,0,0]]).T
	T2 = np.array([[+baselineDistance/2,0,0]]).T
	
	# Rotation Matrix
	R1 = np.identity(3)
	R2 = np.identity(3)

	# Calibration Matrix
	P1 = np.matmul(K1, np.concatenate((R1, -T1.reshape(3,1)), axis=1))
	P2 = np.matmul(K2, np.concatenate((R2, -T2.reshape(3,1)), axis=1))
	
	return P1, P2



def calculate_centroid(cx_decimal, cy_decimal, image):

    # Estimate laser dot diameter
    spot_diameter = laser_diameter + 2 * r * np.tan(laser_divergence / 2)

    # Estimate pixel width
    if 4 <= r <= 16:
        pixel_width = 2 * r * np.tan(AfovPxRad / 2) / 1.5
    else:
        pixel_width = 2 * r * np.tan(AfovPxRad / 2)


    # Generate circle based on point
    # Generate the laser circumference
    x = np.linspace(-spot_diameter / 2, + spot_diameter / 2, 256)
    y = np.sqrt(spot_diameter**2 / 4 - x**2)

    # Generate laser dot
    x = np.hstack((x, x[::-1]))
    y = np.hstack((y, -y))

    # Identify which pixels are illuminated by the laser dots
    x1minPx = np.floor(np.nanmin(x / pixel_width) + cx_decimal)
    x1maxPx = np.ceil(np.nanmax(x / pixel_width) + cx_decimal)
    y1minPx = np.floor(np.nanmin(y / pixel_width) + cy_decimal)
    y1maxPx = np.ceil(np.nanmax(y / pixel_width) + cy_decimal)

    # Identify the meshgrid of pixels
    xPx = np.arange(x1minPx, x1maxPx + 1)
    yPx = np.arange(y1minPx, y1maxPx + 1)
    XPx, YPx = np.meshgrid(xPx, yPx)


    # Calculate centroid
    centroid_x = np.sum(XPx * image) / np.sum(image)
    centroid_y = np.sum(YPx * image) / np.sum(image)

    return centroid_x, centroid_y



if __name__ == "__main__":
	
    # Calculate threshold
    image = np.zeros((100,100))

    noisyimage = add_camera_noise(image, noiseOn=True)

    # Flatten the array
    flattened_data = noisyimage.flatten()

    # Create the histogram
    plt.hist(flattened_data, bins=40, color='blue', alpha=0.7)
    plt.title('Noise interval')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


    mean = np.mean(flattened_data)
    std = np.std(flattened_data)

    threshold = mean + 2*std

    print(threshold)

    # Create the histogram
    plt.hist(flattened_data, bins=40, color='blue', alpha=0.7)
    plt.title('Histogram of Data')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.axvline(x=threshold, color='red', linestyle='--', label='Mean + 2*std')
    plt.legend()
    plt.show()
	

    P1, P2 = generate_calibration_matrix()

    # Read point cloud content
    with open(f'carPC.bin','rb') as pc:
        car_content = np.fromfile(pc, dtype=np.float32).reshape(-1,4)
        car_content = car_content[:, :4]
        pc.close()

    with open(f'pedPC.bin','rb') as pc:
        ped_content = np.fromfile(pc, dtype=np.float32).reshape(-1,4)
        pc.close()

    print(len(car_content))


    x = car_content[:,0]
    y = car_content[:,1]
    z = car_content[:,2]
    colors = []

    for i in range(len(car_content)):
        point = car_content[i]
        if point[3] == 1 and ped_content[i][3] == 0: 
            colors.append('#ff0000')
        elif point[3] == 0 and ped_content[i][3] == 1:
            colors.append('#0000ff')
        else:
            colors.append('#000000')


    fig1 = plt.figure(figsize=(6*2,4*1))
    plt.subplot(1,2,1)
    plt.scatter(x, y, s=0.005, c=colors)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1,2,2)
    plt.scatter(y,z, s=0.005, c=colors)
    plt.xlabel('y')
    plt.ylabel('z')

    plt.tight_layout()
    plt.show()


    # Load noise values
    with open("2point_no_noise_images.pkl", "rb") as f:
        TwoPoint_no_noise_images = pickle.load(f)
        f.close()

    with open("no_noise_images.pkl", "rb") as f:
        no_noise_images = pickle.load(f)
        f.close()
            
        
    c1_centroids = []
    c2_centroids = []
    colorss = []

    i = 0

    for point in car_content:

        if np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2]):
            continue

        # Rearrange point to fit to camera
        point = np.array([point[1], point[2], point[0]]).T
        point = np.hstack((point, np.ones((1,))))

        point -= dotTranslation

        # Cameras projetion coordinates
        x1 = np.matmul(P1, point)
        x2 = np.matmul(P2, point)

        # Normalization
        if x1[2] != 0:
            x1 /= x1[2]

        if x2[2] != 0:
            x2 /= x2[2]


        # Left Camera
        # Remove the integer part of the x coordinate
        cx1_px = x1[0] / pxWidth

        cx1_shift = np.round(cx1_px)

        cx1_decimal = np.round(cx1_px - cx1_shift, 1)

        if cx1_decimal > 0.5:
            cx1_decimal -= 1
            cx1_shift -= 1							
        elif cx1_decimal < -0.5:
            cx1_decimal += 1
            cx1_shift += 1


        # Remove the integer part of the y coordinate
        cy1_px = x1[1] / pxWidth

        cy1_shift = np.round(cy1_px)

        cy1_decimal = np.round(cy1_px - cy1_shift, 1)

        if cy1_decimal > 0.5:
            cy1_decimal -= 1
            cy1_shift -= 1							
        elif cy1_decimal < -0.5:
            cy1_decimal += 1
            cy1_shift += 1

        
        # Right Camera
        # Remove the integer part of the x coordinate
        cx2_px = x2[0] / pxWidth

        cx2_shift = np.round(cx2_px)

        cx2_decimal = np.round(cx2_px - cx2_shift, 1)

        if cx2_decimal > 0.5:
            cx2_decimal -= 1
            cx2_shift -= 1							
        elif cx2_decimal < -0.5:
            cx2_decimal += 1
            cx2_shift += 1


        # Remove the integer part of the y coordinate
        cy2_px = x2[1] / pxWidth

        cy2_shift = np.round(cy2_px)

        cy2_decimal = np.round(cy2_px - cy2_shift, 1)

        if cy2_decimal > 0.5:
            cy2_decimal -= 1
            cy2_shift -= 1							
        elif cy2_decimal < -0.5:
            cy2_decimal += 1
            cy2_shift += 1


        # Calculate distance
        r = np.round(np.sqrt((point[0] ** 2) + (point[1] ** 2) + (point[2] ** 2)))

        if r % 2 != 0:
            r += 1


        if r < overlap_distance:

            colorss.append(colors[i])

            i = i+1


            # Retrieve the respective image
            image1 = np.array(no_noise_images[cx1_decimal][cy1_decimal][r])

            # Apply noise function to image
            noisyImage1 = add_camera_noise(image1, noiseOn=True)

            # Create a boolean mask where True represents values higher than the threshold
            mask = noisyImage1 > threshold

            # Count the number of True values in the mask
            count = np.sum(mask)

            if count == 0:
                continue

            else:

                # Calculate centroid
                centroid_x1, centroid_y1 = calculate_centroid(cx1_decimal, cy1_decimal, noisyImage1)

                # Calculate error
                x_error = centroid_x1 - cx1_decimal
                y_error = centroid_y1 - cy1_decimal

                # Apply error
                c1_cx = (np.round(x1[0] / pxWidth) + x_error) * pxWidth
                c1_cy = (np.round(x1[1] / pxWidth) + y_error) * pxWidth

                c1_image1 = np.array([c1_cx, c1_cy])


            # Retrieve the respective image
            image2 = np.array(no_noise_images[cx2_decimal][cy2_decimal][r])

            # Apply noise function to image
            noisyImage2 = add_camera_noise(image2, noiseOn=True)

            # Create a boolean mask where True represents values higher than the threshold
            mask = noisyImage2 > threshold

            # Count the number of True values in the mask
            count = np.sum(mask)

            if count == 0:
                continue

            else:

                # Calculate centroid
                centroid_x2, centroid_y2 = calculate_centroid(cx2_decimal, cy2_decimal, noisyImage2)

                # Calculate error
                x_error = centroid_x2 - cx2_decimal
                y_error = centroid_y2 - cy2_decimal

                # Apply error
                c2_cx = (np.round(x2[0] / pxWidth) + x_error) * pxWidth
                c2_cy = (np.round(x2[1] / pxWidth) + y_error) * pxWidth

                c2_image1 = np.array([c2_cx, c2_cy])






            # REPEAT THE PROCESS FOR OTHER DOT

            # Add translation coordinates to create the new point 
            point += [x * 2 for x in dotTranslation]

            # Cameras projetion coordinates
            x1 = np.matmul(P1, point)
            x2 = np.matmul(P2, point)

            # Normalization
            if x1[2] != 0:
                x1 /= x1[2]

            if x2[2] != 0:
                x2 /= x2[2]


            # Calculate distance
            r = np.round(np.sqrt((point[0] ** 2) + (point[1] ** 2) + (point[2] ** 2)))

            if r % 2 != 0:
                r += 1


            # Left Camera
            # Remove the integer part of the x coordinate
            cx1_px = x1[0] / pxWidth

            cx1_shift = np.round(cx1_px)

            cx1_decimal = np.round(cx1_px - cx1_shift, 1)

            if cx1_decimal > 0.5:
                cx1_decimal -= 1
                cx1_shift -= 1							
            elif cx1_decimal < -0.5:
                cx1_decimal += 1
                cx1_shift += 1


            # Remove the integer part of the y coordinate
            cy1_px = x1[1] / pxWidth

            cy1_shift = np.round(cy1_px)

            cy1_decimal = np.round(cy1_px - cy1_shift, 1)

            if cy1_decimal > 0.5:
                cy1_decimal -= 1
                cy1_shift -= 1							
            elif cy1_decimal < -0.5:
                cy1_decimal += 1
                cy1_shift += 1


            # Retrieve the respective image
            image3 = np.array(no_noise_images[cx1_decimal][cy1_decimal][r])

            # Apply noise function to image
            noisyImage3 = add_camera_noise(image3, noiseOn=True)

            # Create a boolean mask where True represents values higher than the threshold
            mask = noisyImage3 > threshold

            # Count the number of True values in the mask
            count = np.sum(mask)

            if count == 0:
                continue

            else:

                # Calculate centroid
                centroid_x1, centroid_y1 = calculate_centroid(cx1_decimal, cy1_decimal, noisyImage3)

                # Calculate error
                x_error = centroid_x1 - cx1_decimal
                y_error = centroid_y1 - cy1_decimal

                # Apply error
                c1_cx = (np.round(x1[0] / pxWidth) + x_error) * pxWidth
                c1_cy = (np.round(x1[1] / pxWidth) + y_error) * pxWidth

                c1_image2 = np.array([c1_cx, c1_cy])



            # Right Camera
            # Remove the integer part of the x coordinate
            cx2_px = x2[0] / pxWidth

            cx2_shift = np.round(cx2_px)

            cx2_decimal = np.round(cx2_px - cx2_shift, 1)

            if cx2_decimal > 0.5:
                cx2_decimal -= 1
                cx2_shift -= 1							
            elif cx2_decimal < -0.5:
                cx2_decimal += 1
                cx2_shift += 1


            # Remove the integer part of the y coordinate
            cy2_px = x2[1] / pxWidth

            cy2_shift = np.round(cy2_px)

            cy2_decimal = np.round(cy2_px - cy2_shift, 1)

            if cy2_decimal > 0.5:
                cy2_decimal -= 1
                cy2_shift -= 1							
            elif cy2_decimal < -0.5:
                cy2_decimal += 1
                cy2_shift += 1


            # Retrieve the respective image
            image4 = np.array(no_noise_images[cx2_decimal][cy2_decimal][r])

            # Apply noise function to image
            noisyImage4 = add_camera_noise(image4, noiseOn=True)

            # Create a boolean mask where True represents values higher than the threshold
            mask = noisyImage4 > threshold

            # Count the number of True values in the mask
            count = np.sum(mask)

            if count == 0:
                continue

            else:

                # Calculate centroid
                centroid_x2, centroid_y2 = calculate_centroid(cx2_decimal, cy2_decimal, noisyImage4)

                # Calculate error
                x_error = centroid_x2 - cx2_decimal
                y_error = centroid_y2 - cy2_decimal

                # Apply error
                c2_cx = (np.round(x2[0] / pxWidth) + x_error) * pxWidth
                c2_cy = (np.round(x2[1] / pxWidth) + y_error) * pxWidth

                c2_image2 = np.array([c2_cx, c2_cy])



            c1_centroid = (c1_image1 + c1_image2) / 2
            c2_centroid = (c2_image1 + c2_image2) / 2

            c1_centroids.append(c1_centroid)
            c2_centroids.append(c2_centroid)



        if r >= overlap_distance:

            colorss.append(colors[i])

            i = i+1


            # Retrieve the respective image
            image5 = np.array(TwoPoint_no_noise_images[cx1_decimal][cy1_decimal][r])

            # Apply noise function to image
            noisyImage5 = add_camera_noise(image5, noiseOn=True)

            # Create a boolean mask where True represents values higher than the threshold
            mask = noisyImage5 > threshold

            # Count the number of True values in the mask
            count = np.sum(mask)

            if count == 0:
                continue

            else:

                # Estimate laser dot diameter
                spot_diameter = laser_diameter + 2 * r * np.tan(laser_divergence / 2)

                # Estimate pixel width
                pixel_width = 2 * r * np.tan(AfovPxRad / 2)


                # Left side point
                # Generate circle based on point
                # Generate the laser circumference
                x = np.linspace(-(spot_diameter / 2), +(spot_diameter / 2), 256)	# 256 points
                y = np.sqrt((spot_diameter**2 / 4) - x**2)
                x = x - dotTranslation[0]/2

                x = x + 0
                y = y + 0

                # Generate laser dot
                x = np.hstack((x, x[::-1]))
                y = np.hstack((y, -y))

                # Identify which pixels are illuminated by the laser dots
                x1minPx = np.floor(np.nanmin(x / pixel_width) + cx1_decimal)
                x1maxPx = np.ceil(np.nanmax(x / pixel_width) + cx1_decimal)
                y1minPx = np.floor(np.nanmin(y / pixel_width) + cy1_decimal)
                y1maxPx = np.ceil(np.nanmax(y / pixel_width) + cy1_decimal)

                # Identify the meshgrid of pixels
                x_left = np.arange(x1minPx, x1maxPx + 1)
                y_left = np.arange(y1minPx, y1maxPx + 1)



                # Right side point
                # Generate circle based on point
                # Generate the laser circumference
                x = np.linspace(-(spot_diameter / 2), +(spot_diameter / 2), 256)	# 256 points
                y = np.sqrt((spot_diameter**2 / 4) - x**2)
                x = x + dotTranslation[0]/2

                x = x + 0
                y = y + 0

                # Generate laser dot
                x = np.hstack((x, x[::-1]))
                y = np.hstack((y, -y))

                # Identify which pixels are illuminated by the laser dots
                x1minPx = np.floor(np.nanmin(x / pixel_width) + cx1_decimal)
                x1maxPx = np.ceil(np.nanmax(x / pixel_width) + cx1_decimal)
                y1minPx = np.floor(np.nanmin(y / pixel_width) + cy1_decimal)
                y1maxPx = np.ceil(np.nanmax(y / pixel_width) + cy1_decimal)

                # Identify the meshgrid of pixels
                x_right = np.arange(x1minPx, x1maxPx + 1)
                y_right = np.arange(y1minPx, y1maxPx + 1)


                # Normalize x and y coordinates (to avoid negative indexes)
                x_leftNorm = x_left + abs(min(x_left))
                x_rightNorm = x_right + abs(min(x_left))
                y_leftNorm = y_left + abs(min(y_left))
                y_rightNorm = y_right + abs(min(y_right))

                # Create the final image
                Xfinal, Yfinal = np.meshgrid(np.arange(min(x_leftNorm), max(x_rightNorm)+1), np.arange(y_leftNorm[0], y_leftNorm[-1]+1))

                # Calculate centroid
                centroid_x1 = np.sum(Xfinal * noisyImage5) / np.sum(noisyImage5)
                centroid_y1 = np.sum(Yfinal * noisyImage5) / np.sum(noisyImage5)

                # Calculate error
                x_error = centroid_x1 - cx1_decimal
                y_error = centroid_y1 - cy1_decimal

                # Apply error
                c1_cx = (np.round(x1[0] / pxWidth) + x_error) * pxWidth
                c1_cy = (np.round(x1[1] / pxWidth) + y_error) * pxWidth

                c1_centroids.append(np.array([c1_cx, c1_cy]))



            # Retrieve the respective image
            image6 = np.array(TwoPoint_no_noise_images[cx2_decimal][cy2_decimal][r])

            # Apply noise function to image
            noisyImage6 = add_camera_noise(image6, noiseOn=True)

            # Create a boolean mask where True represents values higher than the threshold
            mask = noisyImage6 > threshold

            # Count the number of True values in the mask
            count = np.sum(mask)

            if count == 0:
                continue

            else:

                # Estimate laser dot diameter
                spot_diameter = laser_diameter + 2 * r * np.tan(laser_divergence / 2)

                # Estimate pixel width
                pixel_width = 2 * r * np.tan(AfovPxRad / 2)


                # Left side point
                # Generate circle based on point
                # Generate the laser circumference
                x = np.linspace(-(spot_diameter / 2), +(spot_diameter / 2), 256)	# 256 points
                y = np.sqrt((spot_diameter**2 / 4) - x**2)
                x = x - dotTranslation[0]/2

                x = x + 0
                y = y + 0

                # Generate laser dot
                x = np.hstack((x, x[::-1]))
                y = np.hstack((y, -y))

                # Identify which pixels are illuminated by the laser dots
                x1minPx = np.floor(np.nanmin(x / pixel_width) + cx2_decimal)
                x1maxPx = np.ceil(np.nanmax(x / pixel_width) + cx2_decimal)
                y1minPx = np.floor(np.nanmin(y / pixel_width) + cy2_decimal)
                y1maxPx = np.ceil(np.nanmax(y / pixel_width) + cy2_decimal)

                # Identify the meshgrid of pixels
                x_left = np.arange(x1minPx, x1maxPx + 1)
                y_left = np.arange(y1minPx, y1maxPx + 1)



                # Right side point
                # Generate circle based on point
                # Generate the laser circumference
                x = np.linspace(-(spot_diameter / 2), +(spot_diameter / 2), 256)	# 256 points
                y = np.sqrt((spot_diameter**2 / 4) - x**2)
                x = x + dotTranslation[0]/2

                x = x + 0
                y = y + 0

                # Generate laser dot
                x = np.hstack((x, x[::-1]))
                y = np.hstack((y, -y))

                # Identify which pixels are illuminated by the laser dots
                x1minPx = np.floor(np.nanmin(x / pixel_width) + cx2_decimal)
                x1maxPx = np.ceil(np.nanmax(x / pixel_width) + cx2_decimal)
                y1minPx = np.floor(np.nanmin(y / pixel_width) + cy2_decimal)
                y1maxPx = np.ceil(np.nanmax(y / pixel_width) + cy2_decimal)

                # Identify the meshgrid of pixels
                x_right = np.arange(x1minPx, x1maxPx + 1)
                y_right = np.arange(y1minPx, y1maxPx + 1)


                # Normalize x and y coordinates (to avoid negative indexes)
                x_leftNorm = x_left + abs(min(x_left))
                x_rightNorm = x_right + abs(min(x_left))
                y_leftNorm = y_left + abs(min(y_left))
                y_rightNorm = y_right + abs(min(y_right))

                # Create the final image
                Xfinal, Yfinal = np.meshgrid(np.arange(min(x_leftNorm), max(x_rightNorm)+1), np.arange(y_leftNorm[0], y_leftNorm[-1]+1))

                # Calculate centroid
                centroid_x2 = np.sum(Xfinal * noisyImage6) / np.sum(noisyImage6)
                centroid_y2 = np.sum(Yfinal * noisyImage6) / np.sum(noisyImage6)

                # Calculate error
                x_error = centroid_x2 - cx2_decimal
                y_error = centroid_y2 - cy2_decimal

                # Apply error
                c2_cx = (np.round(x2[0] / pxWidth) + x_error) * pxWidth
                c2_cy = (np.round(x2[1] / pxWidth) + y_error) * pxWidth

                c2_centroids.append(np.array([c2_cx, c2_cy]))


    c1_centroids = np.array(c1_centroids)
    c2_centroids = np.array(c2_centroids)

    # Call openCV function
    image3D = openCV.triangulatePoints(P1, P2, c1_centroids.T, c2_centroids.T)

    # Denormalization
    new_pointcloud = image3D[0:3,:] / image3D[3,:]



    # Plot the point cloud
    fig2 = plt.figure(figsize=(6*2,4*1))
    plt.subplot(1,2,1)  
    plt.scatter(new_pointcloud[2,:], new_pointcloud[0,:], s=0.025, c = colorss)
    plt.xlabel('z')
    plt.ylabel('x')
    plt.title('UC4.2')

    plt.subplot(1,2,2)
    plt.scatter(new_pointcloud[0,:], new_pointcloud[1,:], s=0.025, c = colorss)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()