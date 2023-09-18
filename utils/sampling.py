import os
import numpy as np
import trimesh
from trimesh.sample import sample_surface
from trimesh.proximity import signed_distance
import torch

from typing import Tuple, List

class MeshSampler:
    def __init__(self, mesh:trimesh.Trimesh, n_points:int=500000, dist_stdv:List[float]=[0.005**0.5, 0.0005**0.5]) -> None:
        self.mesh = mesh
        self.n_points = n_points
        self.dist_stdv = dist_stdv

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:        
        points = self._sample_uniform(int(self.n_points*0.05))
        for sigma in self.dist_stdv:
            points = np.concatenate((points, self._sample_surface(sigma, int(self.n_points*0.95/len(self.dist_stdv)))), axis=0)

        sdf_values = -signed_distance(self.mesh, points)  # (n_points,)

        # filter out any NaN SDF values
        mask = np.isnan(sdf_values)
        points = points[~mask]
        sdf_values = sdf_values[~mask]

        return points, sdf_values
    
    def _sample_uniform(self, n_points) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, (n_points, 3))
    
    def _sample_surface(self, sigma, n_points) -> np.ndarray:
        points, _ = sample_surface(self.mesh, n_points)  # (n_points, 3)
        points += np.random.normal(0.0, sigma, size=points.shape)  # add random noise to offset points from surface
        return points
    

class SDF_Dataset(torch.utils.data.Dataset):
    def __init__(self, sdf_paths:List[str], n_points:int=None, device:str='cpu'):
        self.sdf_paths = sdf_paths
        self.n_points = n_points
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
        if self.n_points is not None:
            assert self.n_points <= point_sdf_pairs.shape[0], "Can't select more than available number of samples"
            point_sdf_pairs = point_sdf_pairs[np.random.choice(point_sdf_pairs.shape[0], self.n_points, replace=False)]
        samples = torch.from_numpy(point_sdf_pairs).to(self.device).to(torch.float32)
        return samples, torch.full((samples.shape[0],1), idx, device=self.device)

    def load_samples(self, path:str) -> np.ndarray:
        with open(os.path.join(path, "data.npy"), 'rb') as file:
            point_sdf_pairs = np.load(file)  # (n_points, 4)

        return point_sdf_pairs