import os
import yaml
import json
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from utils.sampling import MeshSampler

def get_args():
    parser = argparse.ArgumentParser(description="ShapeNet v2 data preprocessing")
    parser.add_argument(
        "--config",
        dest="config",
        required=True,
        help="Path to config file"
    )

    parser.add_argument(
        "--visualize", "-v",
        dest="visualize",
        action="store_true",
        help="Flag for visualization of samples"
    )

    args = parser.parse_args()
    return args

def get_mesh_sdf_samples(mesh_path, n_samples):
    # Load mesh from provided path and sample points
    mesh = trimesh.load(mesh_path, force='mesh', skip_materials=True)
    mesh_sampler = MeshSampler(mesh, n_points=n_samples)
    points, sdf_values = mesh_sampler.get_samples()
    pairs = np.concatenate((points, sdf_values[:,None]), axis=1)  # join [x y z] and SDF together
    return pairs

def main(args, visualize):
    with open(args['train_test_split'], 'r') as file:
        paths = json.load(file)
        training_paths = [os.path.join(args['shapenet_dir'], path, "models/model_normalized.obj") for path in paths['train']]
        test_paths = [os.path.join(args['shapenet_dir'], path, "models/model_normalized.obj") for path in paths['test']]

    for mesh_path in tqdm(training_paths+test_paths):
        subdir = mesh_path.split('/')
        save_path = os.path.join(args['data_dir'], subdir[-3])
        os.makedirs(save_path, exist_ok=True)

        # Generate SDF samples (and corresponding 3D coordinates)
        point_sdf_pairs = get_mesh_sdf_samples(mesh_path, args['n_samples'])

        pos_idxs = point_sdf_pairs[:,3] >= 0
        
        # Save positive and negative SDF values separately (to get more even sampling later)
        with open(os.path.join(save_path, 'pos_sdf.npy'), 'wb') as file:
            np.save(file, point_sdf_pairs[pos_idxs])
        
        with open(os.path.join(save_path, 'neg_sdf.npy'), 'wb') as file:
            np.save(file, point_sdf_pairs[~pos_idxs])

        if visualize:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.scatter(point_sdf_pairs[:,0], point_sdf_pairs[:,1], point_sdf_pairs[:,2], c=point_sdf_pairs[:,3], cmap="plasma", s=30)
            plt.show()
        
if __name__ == "__main__":
    args = get_args()
    with open(args.config, 'r') as file:
        config_args = yaml.safe_load(file)

    main(config_args, args.visualize)