import numpy as np
d = np.load("data/navvis_house2_gaussians_demo.npz")
for k in d.files:
    print(k, d[k].shape, d[k].dtype)
