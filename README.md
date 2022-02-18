## About
PyTorch code for our paper:[**"On Success and Simplicity: A Second Look at Transferable Targeted Attacks"**](http://arxiv.org/abs/2012.11207).
<br> Zhengyu Zhao, Zhuoran Liu, Martha Larson. **NeurIPS 2021**.

**TL;DR:** We demonstrate that the conventional simple, iterative attacks can actually achieve even higher *targeted* transferability than the SOTA, resource-intensive attacks when *enough iterations* are used to ensure convergence and the widely-used Cross-Entropy loss is replaced by a simpler *Logit loss* to prevent the decreasing gradient problem.


### Requirements
torch>=1.7.0; torchvision>=0.8.1; tqdm>=4.31.1; pillow>=7.0.0; matplotlib>=3.2.2;  numpy>=1.18.1; 

### Dataset

The 1000 images from the NIPS 2017 ImageNet-Compatible dataset are provided in the folder ```dataset/images```, along with their metadata in  ```dataset/images.csv```. More details about this dataset can be found in its [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset).

### Evaluation
We evaluated three simple transferable targeted attacks (CE, Po+Trip, and Logit) in the following six transfer scenarios.
If not mentioned specifically, all attacks are integrated with TI, MI, and DI, and run with 300 iterations to ensure convergence.
L<sub>&infin;</sub>=16 is applied.

#### ```eval_single.py```: Single-model transfer.
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_single.PNG" width='700'>
</p>

#### ```eval_ensemble.py```: Ensemble transfer. 
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_ensemble.PNG" width='700'>
</p>

#### ```eval_low_ranked.py```: Transfer with low-ranked targets. 
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_low_ranked.PNG" width='400'>
</p>

#### ```eval_10_targets.py```: "10-Targets (all-source)" setting to compare with the resource-intensive SOTA, [**TTP**](https://arxiv.org/abs/2103.14641), which is based on training target-class-specific GANs. 
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_10_targets.PNG" width='400'>
</p>

#### ```eval_unbounded.py```: Undounded setting to the resource-intensive [**FDA<sup>(N)</sup>+xent**](https://proceedings.neurips.cc//paper/2020/hash/eefc7bfe8fd6e2c8c01aa6ca7b1aab1a-Abstract.html), which is based on training target-class-specific auxiliary binary classifiers.
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_unbounded.PNG" width='700'>
</p>

#### ```eval_tUAP.py```: Generating targeted UAPs.
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_tUAP.PNG" width='300'>
</p>

#### Attacking the Google Cloud Vision API.
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_gg.PNG" width='700'>
</p>

