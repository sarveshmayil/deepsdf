# deepsdf

This is a re-implementation of the CVPR '19 paper "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation" by Park et al. [Paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html)

## Setup

The python setup uses `poetry` which can be installed using
```
curl -sSL https://install.python-poetry.org | python3 -
```

To install the dependencies with poetry,
```
poetry shell
poetry install
```

## Getting data

This repository is set up to use ShapeNetv2 data, but any `.obj` file should work.

You can use
```
python3 load_shapenet.py --data /path/to/data/directory
```
which will compute ground truth SDF values for sampled 3D points and save them for training/testing.

## Training

Use the training script to train the DeepSDF model
```
python3 train_deep_sdf.py --config /path/to/config/file -v
```
which will save the trained model as well as the learned latent vectors at the specified intervals.

## Reconstruction

Use the reconstruction script to visualize the predicted surface of the object using the trained DeepSDF model
```
# For reconstruction of test objects (also learns latent vectors)
python3 reconstruct.py --model /path/to/trained/model --config /path/to/config/file -v

# For reconstruction of train objects using learned latent vectors
python3 reconstruct.py --model /path/to/trained/model --latent-vec /path/to/learned/latent/vectors --config /path/to/config/file -v
```