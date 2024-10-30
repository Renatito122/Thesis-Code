import numpy as np
import cv2 as openCV
import time
import os
from array import array


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

# Distance between the 2 cameras
baselineDistance = 1.2

# Translation vector between two dots of each set of dots
dotTranslation = [11.7e-3, 0, 0, 0]


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

    # Start timer
    start = int(time.time())

    # Iterate over all the original point clouds
    for frame in sorted(os.listdir("velodyne_1")):

        # Process only even frames
        if int(frame.split(".")[0]) % 2 != 0 or int(frame.split(".")[0]) < 7501:
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
             
         
        new_pointcloud = []

        for point in content:

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

            # Call openCV function
            image3D = openCV.triangulatePoints(P1, P2, x1[0:2], x2[0:2]).reshape((1, 4)).flatten()

            # Denormalization
            if image3D[3] != 0:
                image3D /= image3D[3]

            # Rearrange point to point cloud format
            image3D = [image3D[2], image3D[0], image3D[1]]

            new_pointcloud.append(image3D[0])
            new_pointcloud.append(image3D[1])
            new_pointcloud.append(image3D[2])
            new_pointcloud.append(0.0)



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

            # Call openCV function
            image3D = openCV.triangulatePoints(P1, P2, x1[0:2], x2[0:2]).reshape((1, 4)).flatten()

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
        with open(f"new_velodyne_5/{frame}", "wb") as rpc:

            array("f", new_pointcloud).tofile(rpc)

        
    # Stop timer
    print(f"Took {int(time.time()) - start} seconds")