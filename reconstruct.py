import yaml
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from network.deep_sdf_network import Decoder
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
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform."
    )

    parser.add_argument(
        "--mesh", "-m",
        dest="create_mesh",
        action="store_true",
        help="Flag for creating mesh of objects"
    )

    args = parser.parse_args()
    return args

def save_model(directory, filename, model, epoch=None):
    if epoch is not None:
        save_path = os.path.join(directory, filename+"%d.pth" % (epoch))
    else:
        save_path = os.path.join(directory, filename+".pth")
    torch.save(model.state_dict(), save_path)

def main(trained_model_path, args, save_dir, visualize):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    latent_dim = args['latent_dim']
    network_kwargs = args['network_specs']
    n_epochs = args['reconstruct_epochs']

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "latent_vecs"), exist_ok=True)

    with open(args['train_test_split'], 'r') as file:
        test_paths = json.load(file)['test']
        test_paths = [os.path.join(args['data_dir'], path) for path in test_paths]

    decoder = Decoder(latent_dim, **network_kwargs)
    trained_decoder_state = torch.load(trained_model_path)
    decoder.load_state_dict(trained_decoder_state['model_state_dict'])
    decoder = decoder.to(device)
    decoder.eval()

    l1_loss = torch.nn.L1Loss(reduction="sum")

    for test_path in test_paths:
        latent_vec = torch.normal(mean=torch.zeros((1, latent_dim)), std=0.01).to(device)
        latent_vec.requires_grad = True

        dataset = SDF_Dataset([test_path], device)
        data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

        optimizer = torch.optim.Adam(
            [latent_vec],
            lr=args['reconstruct_lr_schedule']['start']
        )

        for epoch in tqdm(range(n_epochs)):
            optimizer.param_groups[0]['lr'] = args['reconstruct_lr_schedule']['start'] * args['reconstruct_lr_schedule']['decay'] ** (epoch // args['reconstruct_lr_schedule']['interval'])

            for data, _ in data_loader:
                data.requires_grad = False
                data = data.reshape(-1, 4)  # stack inputs into 2D
                points = data[:,:3]  # get [x y z] coordinates
                sdf_true = torch.clamp(data[:,-1], -args['sdf_clamping_dist'], args['sdf_clamping_dist']).unsqueeze(1)  # get ground truth SDF value and clamp between provided values
                n_samples = data.shape[0]

                # chunk data
                points = torch.chunk(points, args['batch_split'])  # (batch_split, n, 3)
                sdf_true = torch.chunk(sdf_true, args['batch_split'])  # (batch_split, n, 1)

                batch_loss = 0.0
                optimizer.zero_grad()
                for i in range(args['batch_split']):
                    inp = torch.cat((latent_vec.expand(points[i].shape[0], 1), points[i]), dim=1)  # stack latent vectors and 3D coordinates

                    sdf_pred = decoder(inp)
                    sdf_pred = torch.clamp(sdf_pred, -args['sdf_clamping_dist'], args['sdf_clamping_dist'])  # clamp predictions to match GT
                    batch_split_loss = l1_loss(sdf_pred, sdf_true[i]) / n_samples

                    if args['latent_vec_regularization']:
                        latent_vec_reg_loss = args['latent_vec_reg_lambda'] * min(1, epoch/100) * torch.sum(torch.norm(latent_vec, p=2)) / n_samples
                        batch_split_loss += latent_vec_reg_loss

                    batch_loss += batch_split_loss
                    batch_split_loss.backward()

                optimizer.step()

        save_model(os.path.join(save_dir, "latent_vecs"), os.path.splitext(os.path.basename(test_path))[0], latent_vec)

if __name__ == "__main__":
    args = get_args()
    with open(args.config, 'r') as file:
        config_args = yaml.safe_load(file)

    main(args.trained_model_path, config_args, args.save_dir, args.visualize)