import numpy as np
import matplotlib.pyplot as plt
from FHT_ErrorAnalyzer import HilbertTransformErrorAnalyzer

My_FHTErrorAnalyzer = HilbertTransformErrorAnalyzer()
NRMSE = My_FHTErrorAnalyzer.error_calculation(num_sim=1000, err_range=[-0.1, 0.1], order=1, number_coef=7,
                                                        f_center=8e9)
porc = np.linspace(-1, 1, 1000)
NRMSE_2d = np.mean(NRMSE, axis=2) # ANNIER -> change this add the center of the gaussian distribution

NRMSE_2d[0, :] = np.convolve(NRMSE_2d[0, :], np.ones(5) / 5, mode='same')
NRMSE_2d[1, :] = np.convolve(NRMSE_2d[1, :], np.ones(5) / 5, mode='same')

# Plot for the first case
plt.figure()
plt.plot(porc, np.convolve(NRMSE[0, :, 0], np.ones(3) / 3, mode='same'))
plt.hlines(10, porc[0], porc[-1], linestyle='--', color='red')
plt.plot(porc, NRMSE_2d[0, :], '--k', linewidth=1.5)
plt.legend(['RMSE Case 1', '', 'Mean RMSE (5000 cases)'])
plt.xlabel('Taps error rate (%)')
plt.ylabel('RMSE')
plt.show()

# Plot for the second case
plt.figure()
plt.plot(porc, np.convolve(NRMSE[1, :, 0], np.ones(3) / 3, mode='same'))
plt.hlines(10, porc[0], porc[-1], linestyle='--', color='red')
plt.plot(porc, NRMSE_2d[1, :], '--k', linewidth=1.5)
plt.legend(['RMSE Case 1', '', 'Mean RMSE (5000 cases)'])
plt.xlabel('Taps error rate (%)')
plt.ylabel('RMSE')
plt.show()
