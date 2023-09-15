import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from utils.sampling import MeshSampler

def get_args():
    parser = argparse.ArgumentParser(description="ShapeNet v2 data preprocessing")
    parser.add_argument(
        "--data", "-d",
        dest="data_dir",
        required=True,
        help="Path to directory with downloaded ShapeNet models"
    )

    parser.add_argument(
        "--visualize", "-v",
        dest="visualize",
        action="store_true",
        help="Flag for visualization of samples"
    )

    args = parser.parse_args()
    return args

def get_mesh_sdf_samples(mesh_path):
    mesh = trimesh.load(mesh_path, file_type='obj', force='mesh', skip_materials=True)
    mesh_sampler = MeshSampler(mesh)
    points, sdf_values = mesh_sampler.get_samples()
    pairs = np.concatenate((points, sdf_values[:,None]), axis=1)  # join [x y z] and SDF together
    return pairs

def main(args):
    script_dir = os.path.dirname(__file__)
    mesh_paths = glob(os.path.join(args.data_dir, '*', 'models/model_normalized.obj'))

    for mesh_path in tqdm(mesh_paths):
        subdir = mesh_path.split('/')
        save_path = os.path.join(script_dir, 'data/sdf', subdir[-4], subdir[-3])
        os.makedirs(save_path, exist_ok=True)

        point_sdf_pairs = get_mesh_sdf_samples(mesh_path)
        with open(os.path.join(save_path, 'data.npy'), 'wb') as file:
            np.save(file, point_sdf_pairs)

        if args.visualize:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.scatter(point_sdf_pairs[:,0], point_sdf_pairs[:,1], point_sdf_pairs[:,2], c=point_sdf_pairs[:,3], cmap="plasma", s=30)
            plt.show()
        
if __name__ == "__main__":
    args = get_args()
    main(args)