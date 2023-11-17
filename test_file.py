import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

# Load the array
NRMSE = np.load('NRMSE1000.npy')

# Get the dimensions
dim1, dim2, dim3 = NRMSE.shape

# Plot for the first dimension
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
x1, y1 = np.meshgrid(np.linspace(-1, 1, 1000), range(dim3))
ax1.plot_surface(x1, y1, NRMSE[0, :, :], cmap='viridis')
ax1.set_title('First Dimension')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Value')

# Plot for the second dimension
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')
x2, y2 = np.meshgrid(np.linspace(-1, 1, 1000), range(dim3))
ax2.plot_surface(x2, y2, NRMSE[1, :, :], cmap='viridis')
ax2.set_title('Second Dimension')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_zlabel('Value')

porc = np.linspace(-1, 1, 1000)
NRMSE_2d = np.mean(NRMSE, axis=1)  # ANNIER -> change this add the center of the gaussian distribution

#NRMSE_2d[0, :] = np.convolve(NRMSE_2d[0, :], np.ones(5) / 5, mode='same')
#NRMSE_2d[1, :] = np.convolve(NRMSE_2d[1, :], np.ones(5) / 5, mode='same')

# Plot for the first case
plt.figure()
#plt.plot(porc, np.convolve(NRMSE[0, :, 10], np.ones(3) / 3, mode='same'))
plt.plot(porc, NRMSE[0, :, 0])
plt.plot(porc, NRMSE_2d[0, :], '--k', linewidth=1.5)
plt.legend(['RMSE Case 1', '', 'Mean RMSE (5000 cases)'])
plt.xlabel('Taps error rate (%)')
plt.ylabel('RMSE')


# Plot for the second case
plt.figure()
#plt.plot(porc, np.convolve(NRMSE[1, :, 10], np.ones(3) / 3, mode='same'))
plt.plot(porc, NRMSE[1, :, 0])
plt.plot(porc, NRMSE_2d[1, :], '--k', linewidth=1.5)
plt.legend(['RMSE Case 1', '', 'Mean RMSE (5000 cases)'])
plt.xlabel('Taps error rate (%)')
plt.ylabel('RMSE')




# Show the plots
plt.show(block=True)

