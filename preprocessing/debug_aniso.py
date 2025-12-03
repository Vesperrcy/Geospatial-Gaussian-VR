import numpy as np

data = np.load("data/navvis_house2_gaussians_demo.npz")
R = data["rotations"]      # (N,9)
S = data["scales"]         # (N,3)

N = R.shape[0]
R = R.reshape(N, 3, 3)

# 1) 检查 R 是否接近正交矩阵：R^T R ≈ I
RtR = np.matmul(R.transpose(0, 2, 1), R)      # (N,3,3)
I = np.eye(3)[None, :, :]
err = np.linalg.norm(RtR - I, axis=(1, 2))

print("Rotation orthogonality error: mean=", err.mean(), "max=", err.max())

# 2) 检查 scale 是否为正，且在 clamp 范围内
print("Scales min:", S.min(), "max:", S.max())