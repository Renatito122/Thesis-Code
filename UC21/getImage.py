import numpy as np
import matplotlib.pyplot as plt
import cv2 as openCV


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
		ped_content = ped_content[:, :4]
		pc.close()


	x = car_content[:,0]
	y = car_content[:,1]
	z = car_content[:,2]
	c = []
	
	for i in range(len(car_content)):
		point = car_content[i]
		if point[3] == 1 and ped_content[i][3] == 0: 
			c.append('#ff0000')
		elif point[3] == 0 and ped_content[i][3] == 1:
			c.append('#0000ff')
		else:
			c.append('#000000')
			

	fig = plt.figure(figsize=(6*2,4*1))
	plt.subplot(1,2,1)
	plt.scatter(x, y, s=0.005, c=c)
	plt.xlabel('x')
	plt.ylabel('y')
	
	plt.subplot(1,2,2)
	plt.scatter(y,z, s=0.005, c=c)
	plt.xlabel('y')
	plt.ylabel('z')
	
	plt.tight_layout()
	plt.show()
	


	pointCloud4D = np.array([car_content[:,1], car_content[:,2], car_content[:,0]]).T
	pointCloud4D = np.hstack((pointCloud4D, np.ones((pointCloud4D.shape[0],1))))


	# Get camera projetion coordinates
	x1 = np.matmul(P1, pointCloud4D.T)
	x2 = np.matmul(P2, pointCloud4D.T)

	# Normalization
	x1 = x1[0:2,:] / x1[2,:]
	x2 = x2[0:2,:] / x2[2,:]

	print(x1)

	# Call openCV function
	XestOpenCV = openCV.triangulatePoints(P1, P2, x1, x2)

	# Denormalization
	points2D = XestOpenCV[0:3,:] / XestOpenCV[3,:]  

	print(points2D)

    # Plot the point cloud
	fig = plt.figure(figsize=(6*2,4*1))
	plt.subplot(1,2,1)  
	plt.scatter(points2D[2,:], points2D[0,:], s=0.025, c = c)
	plt.xlabel('z')
	plt.ylabel('x')
	plt.title('UC2.1')
	
	plt.subplot(1,2,2)
	plt.scatter(points2D[0,:], points2D[1,:], s=0.025, c = c)
	plt.xlabel('x')
	plt.ylabel('y')
	
	plt.tight_layout()
	plt.show()
		
        
		
	

    

