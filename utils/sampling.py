import os
import numpy as np
import trimesh
from trimesh.sample import sample_surface
from trimesh.proximity import signed_distance
import torch

from typing import Tuple, List

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
    

class SDF_Dataset(torch.utils.data.Dataset):
    def __init__(self, sdf_paths:List[str], device:str='cpu'):
        self.sdf_paths = sdf_paths
        self.device = device

    def __len__(self):
        return len(self.sdf_paths)
    
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: index of object

        Returns:
            samples: torch Tensor of shape (N,4) with [x y z sdf]
            idx: torch Tensor of corresponding object index of shape (N,1)
        """
        path = self.sdf_paths[idx]
        point_sdf_pairs = self.load_samples(path)
        samples = torch.from_numpy(point_sdf_pairs).to(self.device)
        return samples, torch.full((samples.shape[0],1), idx, device=self.device)

    def load_samples(self, path:str) -> np.ndarray:
        with open(os.path.join(path, "data.npy")) as file:
            point_sdf_pairs = np.load(file)  # (n_points, 4)

        return point_sdf_pairs