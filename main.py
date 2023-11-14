import numpy as np

coef = np.array([-0.127, -0.212, -0.634, 1, 0.634, 0.212, 0.127])

n = len(coef)
reg_coef = np.arange(-n + 2, n, 2)
reg_coef = np.insert(reg_coef, n // 2, 0)

thita = np.where(coef < 0, np.pi, 0)
tk = reg_coef - thita / (2 * np.pi)

print("tk:", tk)

