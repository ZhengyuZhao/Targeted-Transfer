## About
PyTorch code for our arXiv article:

Zhengyu Zhao, Zhuoran Liu, Martha Larson, [**"On Success and Simplicity: A Second Look at Transferable Targeted Attacks"**](https://arxiv.org/abs/1911.02466), 2020.
<!-- <p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure1.PNG" width='600'>
</p> -->
We take a second look at the transferability of targeted attacks and show that their difficulty has been overestimated due to a blind spot in the conventional evaluation procedures.
Specifically, current work has unreasonably restricted attack optimization to a few iterations.
Here, we show that targeted attacks are slow to converge and improve considerably when evaluated with more iterations.
We also show that a simple logit attack performs surprisingly well, remarkably surpassing more complex losses and even achieving performance
comparable to the state of the art, which requires massive training with sophisticated loss.
Additional experiments on attacking the Google Cloud Vision API and creating targeted Universal Adaversarial Perturbations (UAP) are also conducted.

## Implementation

### Overview

This code contains the implementations of:
 1. Transferable targeted attacks in both the single model transfer setting and the ensemble transfer setting. 
 2. Creating targeted UAP by our new logit attack.
 
### Requirements
torch>=1.1.0; torchvision>=0.3.0; tqdm>=4.31.1; pillow>=5.4.1; matplotlib>=3.0.3;  numpy>=1.16.4; 

### Download data

Run [this official script](https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/dataset/download_images.py) to download the ImageNet-Compatible dataset (1000 images) along with the official metadata (ground truth and specified target classes).

### Experiments
Code for all the experiments with comments can be found in the Jupyter Notebook file ```main.ipynb```.
