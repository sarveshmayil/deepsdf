import yaml
import argparse
import numpy as np
import torch
import os
from tqdm import tqdm

from network.deep_sdf_network import Decoder


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
        required=True,
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

    decoder = Decoder(latent_dim, **network_kwargs).to(device)

    latent_vecs = torch.nn.Embedding(n_objects, latent_dim, max_norm=args['latent_vec_bound'])
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
        optimizer.param_groups[0]['lr'] = args['network_lr_schedule']['start'] * args['network_lr_schedule']['decay'] ** (epoch // args['network_lr_schedule']['interval'])
        optimizer.param_groups[1]['lr'] = args['latent_vec_lr_schedule']['start'] * args['latent_vec_lr_schedule']['decay'] ** (epoch // args['latent_vec_lr_schedule']['interval'])

        # Get data from dataloader
        # preprocess data
        # split into batches, pass through model, evaluate loss, backprop

        optimizer.step()

        if epoch % args['save_freq'] == 0:
            save_model(os.path.join(save_dir, "decoder"), "decoder", decoder, epoch)
            save_model(os.path.join(save_dir, "latent_vecs"), "latent_vecs", latent_vecs, epoch)

        pbar.update(1)

if __name__ == "__main__":
    args = get_args()
    with open(args.config, 'r') as file:
        config_args = yaml.load(file)

    main(config_args, args.save_dir)