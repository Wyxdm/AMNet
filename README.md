# AMNet
This is the official implementation for our NeurIPS 2023 paper "Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation".

<div align="center">

<h1>Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation </h1>

[Yuan Wang](https://scholar.google.com.hk/citations?user=Pge14mcAAAAJ&hl=zh-CN)<sup>1*</sup>, &nbsp; 
Naisong Luo<sup>1*</sup>, &nbsp; 
Tianzhu Zhang<sup>1📧</sup>, &nbsp;

<sup>1</sup>University of Science and Technology of China
</div>

## 🚀 Overview
<div align="center">
<img width="800" alt="image" src="figs/framework.png">
</div>

## 📖 Description

Powered by large-scale pre-training, vision foundation models exhibit significant potential in open-world image understanding. Even though individual models have limited capabilities, 
combining multiple such models properly can lead to positive synergies and unleash their full potential. In this work, we present **Matcher**, which segments anything with one shot 
by integrating an all-purpose feature extraction model and a class-agnostic segmentation model. Naively connecting the models results in unsatisfying performance, e.g., the models tend 
to generate matching outliers and false-positive mask fragments. To address these issues, we design a bidirectional matching strategy for accurate cross-image semantic dense matching 
and a robust prompt sampler for mask proposal generation. In addition, we propose a novel instance-level matching strategy for controllable mask merging. The proposed Matcher method 
delivers impressive generalization performance across various segmentation tasks, all without training. For example, it achieves 52.7% mIoU on COCO-20<sup>i</sup> for one-shot semantic 
segmentation, surpassing the state-of-the-art specialist model by 1.6%. In addition, our visualization results show open-world generality and flexibility on images in the wild.


## 🗓️ TODO
- [ ] Online Demo 
- [ ] Release code and models
