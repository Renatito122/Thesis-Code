import numpy as np
import matplotlib.pyplot as plt
import cv2 as openCV
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
    
            
        
    c1_centroids = []
    c2_centroids = []
    colorss = []

    for point in car_content:

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

        c1_centroids.append(np.array([x1[0],x1[1]]))
        c2_centroids.append(np.array([x2[0],x2[1]]))

        

    for a in range(len(c1_centroids)):
        colorss.append(colors[a])


    for point in car_content:

        # Rearrange point to fit to camera
        point = np.array([point[1], point[2], point[0]]).T
        point = np.hstack((point, np.ones((1,))))

        # REPEAT THE PROCESS FOR OTHER DOT

        # Add translation coordinates to create the new point 
        point += dotTranslation

        # Cameras projetion coordinates
        x1 = np.matmul(P1, point)
        x2 = np.matmul(P2, point)

        # Normalization
        if x1[2] != 0:
            x1 /= x1[2]

        if x2[2] != 0:
            x2 /= x2[2]

        c1_centroids.append(np.array([x1[0],x1[1]]))
        c2_centroids.append(np.array([x2[0],x2[1]]))


    colorss = np.tile(colorss, 2)

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
    plt.title('UC3.1')

    plt.subplot(1,2,2)
    plt.scatter(new_pointcloud[0,:], new_pointcloud[1,:], s=0.025, c = colorss)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()