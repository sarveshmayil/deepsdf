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
        """
        Generate 3D cartesian coordinate points and corresponding SDF values
        for provided mesh.

        Args:
            None

        Returns:
            points: numpy array of shape (N,3) with 3D cartesian coordinates
            sdf_values: numpy array of shape (N,) with SDF values
        """
        # Sample points in multiple distributions
        # First, sample uniformally in the entire box (-1 to +1)
        points = self._sample_uniform(int(self.n_points*0.05))
        # Sample along surface and add random gaussian noise
        for sigma in self.dist_stdv:
            points = np.concatenate((points, self._sample_surface(sigma, int(self.n_points*0.95/len(self.dist_stdv)))), axis=0)

        # Compute SDF values for provided points
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
        noise = np.random.normal(0.0, sigma, size=points.shape)
        points += noise  # add random noise to offset points from surface
        return points
    

class SDF_Dataset(torch.utils.data.Dataset):
    def __init__(self, sdf_paths:List[str], subsample:int=None, device:str='cpu'):
        self.sdf_paths = sdf_paths
        self.subsample = subsample
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
        pos_pairs, neg_pairs = self.load_samples(path)

        if self.subsample is not None:
            pos_pairs = torch.from_numpy(pos_pairs[np.random.choice(pos_pairs.shape[0], int(self.subsample/2))])
            neg_pairs = torch.from_numpy(neg_pairs[np.random.choice(neg_pairs.shape[0], int(self.subsample/2))])
        samples = torch.cat((pos_pairs, neg_pairs), dim=0).to(self.device).to(torch.float32)

        return samples, idx

    def load_samples(self, path:str) -> Tuple[np.ndarray, np.ndarray]:
        with open(os.path.join(path, "pos_sdf.npy"), 'rb') as file:
            pos_point_sdf_pairs = np.load(file)  # (n_pos, 4)

        with open(os.path.join(path, "neg_sdf.npy"), 'rb') as file:
            neg_point_sdf_pairs = np.load(file)  # (n_neg, 4)

        return pos_point_sdf_pairs, neg_point_sdf_pairs