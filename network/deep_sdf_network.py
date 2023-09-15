import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

class Decoder(nn.Module):
    def __init__(self, latent_dim:int, hidden_dims:List[int], latent_in:int=4, dropout:float=0.2, weight_norm:bool=True):
        """
        Args:
            latent_dim: size of latent vector
            hidden_dims: list of hidden dimensions of network
            latent_in: layer index where latent vector will be inserted
            dropout: dropout probability
            weight_norm: network weight normalization
        """
        super(Decoder, self).__init__()

        # Add object code to [x y z] input, output single SDF value
        hidden_dims = [latent_dim + 3] + hidden_dims + [1]

        self.latent_in = latent_in
        self.n_layers = len(hidden_dims)
        self.dropout_prob = dropout
        self.relu = nn.ReLU()

        for i in range(0, self.n_layers-1):
            in_dim = hidden_dims[i]
            if i + 1 == latent_in:
                out_dim = hidden_dims[i+1] - hidden_dims[0]
            else:
                out_dim = hidden_dims[i+1]
            
            if weight_norm:
                setattr(self, "linear"+str(i), nn.utils.weight_norm(nn.Linear(in_dim, out_dim)))
            else:
                setattr(self, "linear"+str(i), nn.Linear(in_dim, out_dim))

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Torch tensor of shape (B,L+3) where B=batch size, L=latent dim
                   Last 3 values are [x y z] query coordinates

        Output:
            out: Torch tensor of shape (B,) with SDF predictions
        """
        out = input
        
        for i in range(0, self.n_layers-1):
            out = getattr(self, "linear"+str(i))(out)  # pass through ith layer
            
            if i + 1 == self.latent_in:
                out = torch.cat((out, input), 1)  # stack latent vector and query
            if i < self.n_layers-2:
                out = self.relu(out)
                out = F.dropout(out, p=self.dropout_prob, training=self.training)

        return out

        