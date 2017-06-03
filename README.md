# 3D Colorization
This repository contains implementations of several algorithms for 3d volume colorization. To run the code, you should have the tensorflow installed in your python.

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
Variational Autoencoder. To run:

    cd src-vae/vae
    python train.py

### began
<a href="https://arxiv.org/abs/1703.10717">Boundary Equibilibrium GAN</a>. To run:

    cd src64/began
    python train.py
    
### hourglass
Meomory limitations issue. To run:

    cd src-vae
    python train.py


