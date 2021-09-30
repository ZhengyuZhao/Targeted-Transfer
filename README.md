## About
PyTorch code for our paper:

[**"On Success and Simplicity: A Second Look at Transferable Targeted Attacks"**](http://arxiv.org/abs/2012.11207). Zhengyu Zhao, Zhuoran Liu, Martha Larson. **NeurIPS 2021**.
<!-- <p align="center">
  <img src="https://github.com/ZhengyuZhao/color_adversarial/blob/master/figures/figure1.PNG" width='600'>
</p> -->
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

The 1000 images of the ImageNet-Compatible dataset are provided in the folder ```dataset/images```, along with their metadata in  ```dataset/images.csv```, including the URLs, cropping bounding boxes, classification labels, etc. More details on this dataset can be found in its [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset).

### Evaluation
This code contains the implementations of three simple transferable targeted attacks (CE, Po+Trip, and Logit) in the following six settings.
For all experiments, L<sub>&infin;</sub>=16 is applied.

#### Single-model transfer in ```eval_single.py```
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_single.PNG" width='700'>
</p>

#### Ensemble transfer in ```eval_ensemble.py```
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_ensemble.PNG" width='700'>
</p>

#### A worse-case transfer setting with low-ranked targets in ```eval_low_ranked.py```
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_low_ranked.PNG" width='400'>
</p>

#### "10-Targets" setting to compare with TTP, which is based on training target class-specific GANs, in ```eval_10_targets.py```
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_10_targets.PNG" width='400'>
</p>

#### Undounded setting to compare with FDA, which is based on training target class-specific auxiliary binary classifier(s), in ```eval_unbounded.py```
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_unbounded.PNG" width='700'>
</p>

#### Generating targeted UAPs in ```eval_tUAP.py``` 
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_tUAP.PNG" width='300'>
</p>

#### Additional results on attacking the Google Cloud Vision API
<p align="left">
  <img src="https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/Figures/transfer_gg.PNG" width='700'>
</p>

