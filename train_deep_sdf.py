import yaml
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from network.deep_sdf_network import Decoder
from utils.sampling import SDF_Dataset


def get_args():
    parser = argparse.ArgumentParser(description="Training Deep SDF")
    parser.add_argument(
        "--config",
        dest="config",
        required=True,
        help="Path to config file"
    )

    parser.add_argument(
        "--output", "-o",
        dest="save_dir",
        default=os.path.join(os.path.dirname(__file__), "results", "checkpoints"),
        help="Path to directory where checkpoints are saved"
    )

    parser.parse_args()

def save_model(directory, filename, model, epoch):
    torch.save(model.state_dict(), os.path.join(directory, filename+"%d.pth" % (epoch)))

def main(args, save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = args['latent_dim']
    network_kwargs = args['network_specs']
    n_epochs = args['epochs']

    os.makedirs(save_dir, exist_ok=True)

    with open(args['train_test_split'], 'r') as file:
        training_paths = json.load(file)['train']
        training_paths = [os.path.join(args['data_dir'], path) for path in training_paths]

    dataset = SDF_Dataset(training_paths, device)
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

    decoder = Decoder(latent_dim, **network_kwargs).to(device)

    latent_vecs = torch.nn.Embedding(len(dataset), latent_dim, max_norm=args['latent_vec_bound'], device=device)
    torch.nn.init.normal_(latent_vecs, mean=0.0, std=1.0 / np.sqrt(latent_dim))

    l1_loss = torch.nn.L1Loss(reduction="sum")

    optimizer = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": args['network_lr_schedule']['start']
            },
            {
                "params": latent_vecs.parameters(),
                "lr": args['latent_vec_lr_schedule']['start']
            }
        ]
    )

    decoder.train()
    pbar = tqdm(total=n_epochs, desc="Deep SDF Training")
    for epoch in range(n_epochs):
        # Updates learning rates according to specified 
        optimizer.param_groups[0]['lr'] = args['network_lr_schedule']['start'] * args['network_lr_schedule']['decay'] ** (epoch // args['network_lr_schedule']['interval'])
        optimizer.param_groups[1]['lr'] = args['latent_vec_lr_schedule']['start'] * args['latent_vec_lr_schedule']['decay'] ** (epoch // args['latent_vec_lr_schedule']['interval'])

        # Get data from dataloader
        # preprocess data
        # split into batches, pass through model, evaluate loss, backprop

        for data, obj_idxs in data_loader:
            data.requires_grad = False
            points = data[:,:3]  # get [x y z] coordinates
            sdf_true = torch.clamp(data[:,-1], -args['sdf_clamping_dist'], args['sdf_clamping_dist'])  # get ground truth SDF value and clamp between provided values
            n_samples = data.shape[0]

            # chunk data
            points = torch.chunk(points, args['batch_split'])  # (batch_split, n, 3)
            sdf_true = torch.chunk(sdf_true, args['batch_split'])  # (batch_split, n, 1)
            obj_idxs = torch.chunk(obj_idxs, args['batch_split'])  # (batch_split, n, 1)

            batch_loss = 0.0
            optimizer.zero_grad()
            for i in range(args['batch_split']):
                batch_latent_vecs = latent_vecs(obj_idxs[i])
                inp = torch.cat((batch_latent_vecs, points[i]), dim=1)  # stack latent vectors and 3D coordinates

                sdf_pred = decoder(inp)
                sdf_pred = torch.clamp(sdf_pred, -args['sdf_clamping_dist'], args['sdf_clamping_dist'])  # clamp predictions to match GT
                batch_split_loss = l1_loss(sdf_pred, sdf_true[i]) / n_samples

                if args['latent_vec_regularization']:
                    latent_vec_reg_loss = args['latent_vec_reg_lambda'] * min(1, epoch/100) * torch.sum(torch.norm(batch_latent_vecs, p=2, dim=1)) / n_samples
                    batch_split_loss += latent_vec_reg_loss

                batch_loss += batch_split_loss
                batch_split_loss.backward()

        optimizer.step()

        if epoch % args['save_freq'] == 0:
            save_model(os.path.join(save_dir, "decoder"), "decoder", decoder, epoch)
            save_model(os.path.join(save_dir, "latent_vecs"), "latent_vecs", latent_vecs, epoch)

        pbar.update(1)

if __name__ == "__main__":
    args = get_args()
    with open(args.config, 'r') as file:
        config_args = yaml.safe_load(file)

    main(config_args, args.save_dir)