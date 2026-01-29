import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("data/TumTLS_v2.ply")
print("points:", np.asarray(pcd.points).shape)
print("normals:", np.asarray(pcd.normals).shape)
