# Condor Usage

add these lines to your ~/.profile

    alias cdw='cd /scratch/cluster/yzp12'
    alias showq='condor_q -g -run | grep eldar'
    alias showqm='condor_q -submitter yzp12'
    alias sbatch='condor_submit'
    alias scancel='condor_rm'

submit job to condor: 
    
    sbatch condor_submit

show your jobs: 

    showqm

show all jobs: 

    showq

cancel job: 

    scancel jobid

edit condor_submit before your submission!
change eldar-xx in 'condor_submit' depends on current jobs on the machines. Find a machine with no job runnig. Also, eldar 20-29 are better than others.


# VoxelDCGAN

Implementation of a 3D shape generative model based on <a href="https://arxiv.org/abs/1511.06434">deep convolutional generative adversarial nets</a> (DCGAN) with techniques of <a href="https://github.com/openai/improved-gan">improved-gan</a>.

Experimental results on the <a href="http://shapenet.cs.stanford.edu/">ShapeNet</a> dataset are shown below.



