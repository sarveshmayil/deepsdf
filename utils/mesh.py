import torch
from skimage.measure import marching_cubes
from tqdm import tqdm


def create_mesh(decoder, latent_vec, N:int=256, batch_size:int=32768):
    device = latent_vec.device
    decoder.eval()

    # Marching Cubes Algorithm
    voxel_origin = torch.LongTensor([-1, -1, -1])
    voxel_size = 2.0 / (N - 1)

    n_samples = N**3

    overall_index = torch.arange(0, n_samples, 1, out=torch.LongTensor())

    samples = torch.zeros(n_samples, 4)
    samples[:,0] = ((overall_index // N) // N) % N
    samples[:,1] = (overall_index // N) % N
    samples[:,2] = overall_index % N

    samples[:,:3] = samples[:,:3] * voxel_size + voxel_origin
    samples.requires_grad = False

    head = 0
    pbar = tqdm(total=n_samples, desc="Computing SDF grid")
    while head < n_samples:
        sample_subset = samples[head:min(head+batch_size, n_samples), :3]
        inp = torch.cat((latent_vec.expand(sample_subset.shape[0], -1), sample_subset.to(device)), dim=1)  # stack latent vectors and 3D coordinates
        samples[head:min(head+batch_size, n_samples), 3] = decoder(inp).squeeze(1).detach().cpu()
        head += batch_size
        pbar.update(batch_size)
    pbar.close()

    sdf_pred = samples[:,3].reshape(N,N,N)
    sdf_grid = sdf_pred.cpu().numpy()
    # print(sdf_grid.min(), sdf_grid.max())
    # vertices, faces, _, _ = marching_cubes(sdf_grid, level=0.0, spacing=[voxel_size, voxel_size, voxel_size])
    vertices, faces, _, _ = marching_cubes(sdf_grid, spacing=[voxel_size, voxel_size, voxel_size])

    return vertices, faces
        
