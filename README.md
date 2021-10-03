## About
PyTorch code for our **NeurIPS 2021** paper:

[**"On Success and Simplicity: A Second Look at Transferable Targeted Attacks"**](http://arxiv.org/abs/2012.11207). Zhengyu Zhao, Zhuoran Liu, Martha Larson.

> **Abstract**: Achieving transferability of targeted attacks is reputed to be remarkably difficult.
The current state of the art has resorted to resource-intensive solutions that necessitate training model(s) for each target class with additional data.
In our investigation, we find, however, that simple transferable attacks which require neither additional data nor model training can achieve surprisingly strong targeted transferability.
This insight has been overlooked until now, mainly because the widespread practice of attacking with only few iterations has largely limited the attack convergence to optimal targeted transferability.
In particular, we, for the first time, identify that a very simple logit loss can yield even better results than the resource-intensive state of the art.
Our analysis spans a variety of transfer scenarios, especially including three new, realistic scenarios: an ensemble transfer scenario with little model similarity, a worse-case scenario with attacks targeting low-ranked classes, and also a real-world attack on the Google Cloud Vision API.
Results in these new transfer scenarios demonstrate that the commonly adopted, easy scenario cannot fully reveal the actual properties of different attacks and may cause misleading comparisons.
We also show the usefulness of the simple logit loss for generating targeted universal adversarial perturbations in a data-free manner.
Overall, the aim of our analysis is to inspire a more meaningful evaluation on targeted transferability.

### Requirements
torch>=1.7.0; torchvision>=0.8.1; tqdm>=4.31.1; pillow>=7.0.0; matplotlib>=3.2.2;  numpy>=1.18.1; 

### Dataset

The 1000 images from the ImageNet-Compatible dataset are provided in the folder ```dataset/images```, along with their metadata in  ```dataset/images.csv```. More details about this dataset can be found in its [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset).

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

#### ```eval_10_targets.py```: "10-Targets (all-source)" setting to compare with the state-of-the-art [**TTP**](https://arxiv.org/abs/2103.14641), which is based on training target-class-specific GANs. 
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_10_targets.PNG" width='400'>
</p>

#### ```eval_unbounded.py```: Undounded setting to compare with [**FDA<sup>(N)</sup>+xent**](https://proceedings.neurips.cc//paper/2020/hash/eefc7bfe8fd6e2c8c01aa6ca7b1aab1a-Abstract.html), which is based on training target-class-specific auxiliary binary classifiers.
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

