# Condor Usage

submit job to condor: 
    
    condor_submit configure_file

show your jobs: 

    condor_q -submitter username

show all jobs: 

    condor_q -g -run | grep eldar

cancel job: 

    condor_rm jobid

edit condor_submit before your submission!
change eldar-xx in 'condor_submit' depends on current jobs on the machines. Find a machine with no job runnig. Also, eldar 20-29 are better than others.


# VoxelDCGAN

Implementation of a 3D shape generative model based on <a href="https://arxiv.org/abs/1511.06434">deep convolutional generative adversarial nets</a> (DCGAN) with techniques of <a href="https://github.com/openai/improved-gan">improved-gan</a>.

Experimental results on the <a href="http://shapenet.cs.stanford.edu/">ShapeNet</a> dataset are shown below.



