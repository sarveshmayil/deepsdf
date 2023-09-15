import numpy as np
import trimesh
from trimesh.sample import sample_surface
from trimesh.proximity import signed_distance

from typing import Tuple

class MeshSampler:
    def __init__(self, mesh:trimesh.Trimesh, n_points:int=500, dist_stdv:float=0.0005**0.5) -> None:
        self.mesh = mesh
        self.n_points = n_points
        self.dist_stdv = dist_stdv

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        points, _ = sample_surface(self.mesh, self.n_points)  # (n_points, 3)
        points += np.random.normal(0.0, self.dist_stdv, size=points.shape)  # add random noise to offset points from surface
        
        sdf_values = -signed_distance(self.mesh, points)  # (n_points,)

        # filter out any NaN SDF values
        mask = np.isnan(sdf_values)
        points = points[~mask]
        sdf_values = sdf_values[~mask]

        return points, sdf_values