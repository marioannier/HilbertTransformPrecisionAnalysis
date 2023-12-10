import numpy as np
import matplotlib.pyplot as plt
from FHT_ErrorAnalyzer import HilbertTransformErrorAnalyzer

My_FHTErrorAnalyzer = HilbertTransformErrorAnalyzer()
err_range = [-0.025, 0.025]
num_sim = 100
order = 1
number_coef = 7
f_center = 8e9
NRMSE = My_FHTErrorAnalyzer.error_calculation(num_sim, err_range, order, number_coef, f_center)
My_FHTErrorAnalyzer.plot_nrmse(NRMSE, err_range, num_sim)
#My_FHTErrorAnalyzer.plot_3d_nrmse(NRMSE, err_range, num_sim)
