import yaml
import json
import argparse
import torch
import trimesh
import os
from tqdm import tqdm

from network.deep_sdf_network import Decoder
from utils.mesh import create_mesh
from utils.sampling import SDF_Dataset


def get_args():
    parser = argparse.ArgumentParser(description="Reconstruct mesh using trained Deep SDF")
    parser.add_argument(
        "--model", "-m",
        dest="trained_model_path",
        required=True,
        help="Path to trained Deep SDF model (.pth)"
    )

    parser.add_argument(
        "--latent-vec", "-l",
        dest="latent_vec_path",
        default=None,
        help="Path to learned latent vector (.pth)"
    )
    
    parser.add_argument(
        "--config",
        dest="config",
        required=True,
        help="Path to config file"
    )

    parser.add_argument(
        "--output", "-o",
        dest="save_dir",
        default=os.path.join(os.path.dirname(__file__), "results", "reconstructed"),
        help="Path to directory where reconstructed latent codes are saved"
    )

    parser.add_argument(
        "--visualize", "-v",
        dest="visualize",
        action="store_true",
        help="Flag for visualization of reconstructed mesh"
    )

    args = parser.parse_args()
    return args

def save_latent_vec(directory, filename, latent_vec):
    save_path = os.path.join(directory, filename+".pth")
    torch.save(latent_vec, save_path)

def main(trained_model_path, args, save_dir, visualize):
    # Parse provided specs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    latent_dim = args['latent_dim']
    network_kwargs = args['network_specs']
    n_epochs = args['reconstruct_epochs']
    n_subsamples = args['samples_per_scene'] if 'samples_per_scene' in args else None

    # Make save directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "latent_vecs"), exist_ok=True)

    with open(args['train_test_split'], 'r') as file:
        test_paths = json.load(file)['test']
        test_paths = [os.path.join(args['data_dir'], path) for path in test_paths]

    # Create dataset
    dataset = SDF_Dataset(test_paths, n_subsamples, device)

    # Initialize decoder from provided specs and load saved weights
    decoder = Decoder(latent_dim, **network_kwargs)
    decoder.load_state_dict(torch.load(trained_model_path))
    decoder = decoder.to(device)
    decoder.eval()

    l1_loss = torch.nn.L1Loss()

    for i, test_path in enumerate(test_paths):
        # Create latent vector and initialize weights
        latent_vec = torch.normal(mean=torch.zeros((1, latent_dim)), std=0.01).to(device)
        latent_vec.requires_grad = True

        optimizer = torch.optim.Adam(
            [latent_vec],
            lr=args['reconstruct_lr_schedule']['start']
        )

        data, _ = dataset[i]
        data.requires_grad = False
        points = data[:,:3]  # get [x y z] coordinates
        sdf_true = torch.clamp(data[:,-1], -args['sdf_clamping_dist'], args['sdf_clamping_dist']).unsqueeze(1)  # get ground truth SDF value and clamp between provided values

        for epoch in tqdm(range(n_epochs)):
            # Updates learning rate according to specified schedule
            optimizer.param_groups[0]['lr'] = args['reconstruct_lr_schedule']['start'] * args['reconstruct_lr_schedule']['decay'] ** (epoch // args['reconstruct_lr_schedule']['interval'])

            optimizer.zero_grad()
            inp = torch.cat((latent_vec.expand(points.shape[0], -1), points), dim=1)  # stack latent vectors and 3D coordinates

            # forward pass
            sdf_pred = decoder(inp)
            sdf_pred = torch.clamp(sdf_pred, -args['sdf_clamping_dist'], args['sdf_clamping_dist'])  # clamp predictions to match GT
            
            # Compute L1 Loss
            loss = l1_loss(sdf_pred, sdf_true)

            # Compute regularization loss on latent vector
            if args['latent_vec_regularization']:
                latent_vec_reg_loss = args['latent_vec_reg_lambda'] * torch.mean(torch.pow(latent_vec, 2))
                loss += latent_vec_reg_loss

            # Backprop, gradient descent
            loss.backward()
            optimizer.step()

        save_latent_vec(os.path.join(save_dir, "latent_vecs"), os.path.splitext(os.path.basename(test_path))[0], latent_vec)

        # visualize mesh using trimesh and marching cubes algorithm
        if visualize:
            vertices, faces = create_mesh(decoder, latent_vec, N=128)
            scene = trimesh.Scene()
            reconstruct_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            scene.add_geometry(reconstruct_mesh)
            scene.show()

def main_learned_code(trained_model_path, latent_vec_path, args):
    # Parse provided specs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    latent_dim = args['latent_dim']
    network_kwargs = args['network_specs']

    # Initialize decoder from provided specs and load saved weights
    decoder = Decoder(latent_dim, **network_kwargs)
    decoder.load_state_dict(torch.load(trained_model_path))
    decoder = decoder.to(device)
    decoder.eval()

    # Load saved latent vector(s)
    latent_vecs = torch.load(latent_vec_path)['weight']

    if latent_vecs.ndim == 1:
        latent_vecs = latent_vecs.unsqueeze(0)

    # Loop through latent vectors
    for i in range(latent_vecs.shape[0]):
        # visualize mesh using trimesh and marching cubes algorithm
        vertices, faces = create_mesh(decoder, latent_vecs[i], N=128)
        scene = trimesh.Scene()
        reconstruct_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        scene.add_geometry(reconstruct_mesh)
        scene.show()

if __name__ == "__main__":
    args = get_args()
    with open(args.config, 'r') as file:
        config_args = yaml.safe_load(file)

    if args.latent_vec_path is not None:
        main_learned_code(args.trained_model_path, args.latent_vec_path, config_args)
    else:
        main(args.trained_model_path, config_args, args.save_dir, args.visualize)