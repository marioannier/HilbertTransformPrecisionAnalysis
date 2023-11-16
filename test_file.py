import numpy as np

error_limits = np.linspace(-0.1, 0.1, 10)

for i, e in enumerate(error_limits):
    # adding the error
    tk_rand = np.random.rand(7) * e
    print("Simulation number", tk_rand)
