import numpy as np
import matplotlib.pyplot as plt
from FHT_ErrorAnalyzer import HilbertTransformErrorAnalyzer

My_FHTErrorAnalyzer = HilbertTransformErrorAnalyzer()
NRMSE = My_FHTErrorAnalyzer.error_calculation(num_sim=1000, err_range=[-0.0125, 0.0125], order=1, number_coef=7,
                                              f_center=8e9)
np.save('NRMSE1000.npy', NRMSE)
porc = np.linspace(-1, 1, 1000)
NRMSE_2d = np.mean(NRMSE, axis=2)  # ANNIER -> change this add the center of the gaussian distribution

# NRMSE_2d[0, :] = np.convolve(NRMSE_2d[0, :], np.ones(5) / 5, mode='same')
# NRMSE_2d[1, :] = np.convolve(NRMSE_2d[1, :], np.ones(5) / 5, mode='same')

# Plot for the first case
plt.figure()
# plt.plot(porc, np.convolve(NRMSE[0, :, 10], np.ones(3) / 3, mode='same'))
plt.plot(porc, NRMSE[0, :, 0])
plt.plot(porc, NRMSE_2d[0, :], '--k', linewidth=1.5)
plt.legend(['RMSE Case 1', '', 'Mean RMSE (5000 cases)'])
plt.xlabel('Taps error rate (%)')
plt.ylabel('RMSE')
plt.show()

# Plot for the second case
plt.figure()
# plt.plot(porc, np.convolve(NRMSE[1, :, 10], np.ones(3) / 3, mode='same'))
plt.plot(porc, NRMSE[1, :, 0])
plt.plot(porc, NRMSE_2d[1, :], '--k', linewidth=1.5)
plt.legend(['RMSE Case 1', '', 'Mean RMSE (5000 cases)'])
plt.xlabel('Taps error rate (%)')
plt.ylabel('RMSE')
plt.show()
