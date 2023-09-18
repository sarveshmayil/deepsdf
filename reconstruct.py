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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    latent_dim = args['latent_dim']
    network_kwargs = args['network_specs']
    n_epochs = args['reconstruct_epochs']
    n_train_samples = args['n_train_samples'] if 'n_train_samples' in args else None

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "latent_vecs"), exist_ok=True)

    with open(args['train_test_split'], 'r') as file:
        test_paths = json.load(file)['test']
        test_paths = [os.path.join(args['data_dir'], path) for path in test_paths]

    dataset = SDF_Dataset(test_paths, n_train_samples, device)

    decoder = Decoder(latent_dim, **network_kwargs)
    decoder.load_state_dict(torch.load(trained_model_path))
    decoder = decoder.to(device)
    decoder.eval()

    l1_loss = torch.nn.L1Loss()

    for i, test_path in enumerate(test_paths):
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
        print(sdf_true.min(), sdf_true.max())

        for epoch in tqdm(range(n_epochs)):
            optimizer.param_groups[0]['lr'] = args['reconstruct_lr_schedule']['start'] * args['reconstruct_lr_schedule']['decay'] ** (epoch // args['reconstruct_lr_schedule']['interval'])

            optimizer.zero_grad()
            inp = torch.cat((latent_vec.expand(points.shape[0], -1), points), dim=1)  # stack latent vectors and 3D coordinates

            sdf_pred = decoder(inp)
            sdf_pred = torch.clamp(sdf_pred, -args['sdf_clamping_dist'], args['sdf_clamping_dist'])  # clamp predictions to match GT
            loss = l1_loss(sdf_pred, sdf_true)

            if args['latent_vec_regularization']:
                latent_vec_reg_loss = args['latent_vec_reg_lambda'] * torch.mean(torch.pow(latent_vec, 2))
                loss += latent_vec_reg_loss

            loss.backward()
            optimizer.step()

        save_latent_vec(os.path.join(save_dir, "latent_vecs"), os.path.splitext(os.path.basename(test_path))[0], latent_vec)

        if visualize:
            vertices, faces = create_mesh(decoder, latent_vec, N=128)
            scene = trimesh.Scene()
            reconstruct_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            scene.add_geometry(reconstruct_mesh)
            scene.show()

if __name__ == "__main__":
    args = get_args()
    with open(args.config, 'r') as file:
        config_args = yaml.safe_load(file)

    main(args.trained_model_path, config_args, args.save_dir, args.visualize)