# 3D Colorization
This repository contains implementations of several algorithms for 3d volume colorization. Given a binary voxel, the model outputs the same shape with colorizations. To run the code, you should have the tensorflow installed in your python. 

## Dataset
Our model takes the 3d voxel representations as input. We used <a href="https://arxiv.org/abs/1512.03012">Shapenet Core dataset</a> for training. Before training, the conversion from obj format to voxel representations is required and implenmented under the 3d-colorization/objLoader directory. The dataset can't be provided directly due to license limitations. Please contact the Shapenet team for dataset downloading. 

## Algorithms for 3d volume colorization

### cwgan 
<a href="https://arxiv.org/abs/1701.07875">Conditional Wasserstein GAN</a>. To run:

    cd src64/cwgan
    python train.py
    
### cwgan with surface kernel
Still in process. To run:

    cd src64/cwgan-conv2_5d
    python train.py

### vae 
<a href="https://arxiv.org/abs/1606.05908">Variational Autoencoder</a>. To run:

    cd src-vae/vae
    python train.py

### began
<a href="https://arxiv.org/abs/1703.10717">Boundary Equibilibrium GAN</a>. To run:

    cd src64/began
    python train.py
    
### hourglass
<a href="https://arxiv.org/abs/1603.06937">Stacked Hourglass Networks</a>. Only two stacks due to the meomory limitation issue. To run:

    cd src-vae
    python train.py


