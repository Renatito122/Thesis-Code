import numpy as np
import cv2 as openCV
import time
import os
from array import array
import pickle
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

# Distance between the 2 cameras
baselineDistance = 1.2


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


    # Start timer
    start = int(time.time())

    # Iterate over all the original point clouds
    for frame in sorted(os.listdir("velodyne_1")):

        # Process only odd frames
        if int(frame.split(".")[0]) % 2 == 0 or int(frame.split(".")[0]) > 7500:
            continue

        # Identify the current point cloud
        print(f"Processing frame {frame} - {int(time.time())}")

        # Setup ideal calibration matrices
        P1, P2 = generate_calibration_matrix()

        # Read point cloud content
        with open(f"velodyne_1/{frame}", "rb") as pc:
            content = np.fromfile(pc, dtype=np.float32).reshape(-1,4)
            content = content[:, :4]
            pc.close()

        # Load noise values
        with open("no_noise_images.pkl", "rb") as f:
            no_noise_images = pickle.load(f)
            f.close()
             
         
        new_pointcloud = []

        for point in content:

            if np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2]):
                continue

            # Rearrange point to fit to camera
            point = np.array([point[1], point[2], point[0]]).T
            point = np.hstack((point, np.ones((1,))))

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
                # Estimate laser dot diameter
                spot_diameter = laser_diameter + 2 * r * np.tan(laser_divergence / 2)

                # Estimate pixel width
                if 4 <= r <= 16:
                    pixel_width = 2 * r * np.tan(HfovPx / 2) / 1.5
                else:
                    pixel_width = 2 * r * np.tan(HfovPx / 2)


                # Generate circle based on point
                # Generate the laser circumference
                x = np.linspace(-spot_diameter / 2, + spot_diameter / 2, 256)
                y = np.sqrt(spot_diameter**2 / 4 - x**2)

                # Generate laser dot
                x = np.hstack((x, x[::-1]))
                y = np.hstack((y, -y))

                # Identify which pixels are illuminated by the laser dots
                x1minPx = np.floor(np.nanmin(x / pixel_width) + cx1_decimal)
                x1maxPx = np.ceil(np.nanmax(x / pixel_width) + cx1_decimal)
                y1minPx = np.floor(np.nanmin(y / pixel_width) + cy1_decimal)
                y1maxPx = np.ceil(np.nanmax(y / pixel_width) + cy1_decimal)

                # Identify the meshgrid of pixels
                xPx = np.arange(x1minPx, x1maxPx + 1)
                yPx = np.arange(y1minPx, y1maxPx + 1)
                XPx, YPx = np.meshgrid(xPx, yPx)


                # Calculate centroid
                centroid_x1 = np.sum(XPx * noisyImage1) / np.sum(noisyImage1)
                centroid_y1 = np.sum(YPx * noisyImage1) / np.sum(noisyImage1)

                # Calculate error
                x_error = centroid_x1 - cx1_decimal
                y_error = centroid_y1 - cy1_decimal

                # Apply error
                c1_cx = (np.round(x1[0] / pxWidth) + x_error) * pxWidth
                c1_cy = (np.round(x1[1] / pxWidth) + y_error) * pxWidth

                c1_image = np.array([c1_cx, c1_cy])



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
                # Estimate laser dot diameter
                spot_diameter = laser_diameter + 2 * r * np.tan(laser_divergence / 2)

                # Estimate pixel width
                if 4 <= r <= 16:
                    pixel_width = 2 * r * np.tan(HfovPx / 2) / 1.5
                else:
                    pixel_width = 2 * r * np.tan(HfovPx / 2)


                # Generate circle based on point
                # Generate the laser circumference
                x = np.linspace(-spot_diameter / 2, + spot_diameter / 2, 256)
                y = np.sqrt(spot_diameter**2 / 4 - x**2)

                # Generate laser dot
                x = np.hstack((x, x[::-1]))
                y = np.hstack((y, -y))

                # Identify which pixels are illuminated by the laser dots
                x1minPx = np.floor(np.nanmin(x / pixel_width) + cx2_decimal)
                x1maxPx = np.ceil(np.nanmax(x / pixel_width) + cx2_decimal)
                y1minPx = np.floor(np.nanmin(y / pixel_width) + cy2_decimal)
                y1maxPx = np.ceil(np.nanmax(y / pixel_width) + cy2_decimal)

                # Identify the meshgrid of pixels
                xPx = np.arange(x1minPx, x1maxPx + 1)
                yPx = np.arange(y1minPx, y1maxPx + 1)
                XPx, YPx = np.meshgrid(xPx, yPx)


                # Calculate centroid
                centroid_x2 = np.sum(XPx * noisyImage2) / np.sum(noisyImage2)
                centroid_y2 = np.sum(YPx * noisyImage2) / np.sum(noisyImage2)

                # Calculate error
                x_error = centroid_x2 - cx2_decimal
                y_error = centroid_y2 - cy2_decimal

                # Apply error
                c2_cx = (np.round(x2[0] / pxWidth) + x_error) * pxWidth
                c2_cy = (np.round(x2[1] / pxWidth) + y_error) * pxWidth

                c2_image = np.array([c2_cx, c2_cy])



            # Call openCV function
            image3D = openCV.triangulatePoints(P1, P2, c1_image, c2_image).reshape((1, 4)).flatten()

            # Denormalization
            if image3D[3] != 0:
                image3D /= image3D[3]

            # Rearrange point to point cloud format
            image3D = [image3D[2], image3D[0], image3D[1]]

            new_pointcloud.append(image3D[0])
            new_pointcloud.append(image3D[1])
            new_pointcloud.append(image3D[2])
            new_pointcloud.append(0.0)

        # Write point cloud to file
        with open(f"velodyne_2/{frame}", "wb") as rpc:

            array("f", new_pointcloud).tofile(rpc)

        
    # Stop timer
    print(f"Took {int(time.time()) - start} seconds")
