# AMNet
This is the official implementation for our NeurIPS 2023 paper "Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation".

<div align="center">

<h1>Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation </h1>

[Yuan Wang](https://scholar.google.com.hk/citations?user=Pge14mcAAAAJ&hl=zh-CN)<sup>1*</sup>, &nbsp; 
Naisong Luo<sup>1*</sup>, &nbsp; 
Tianzhu Zhang<sup>1📧</sup>, &nbsp;

<sup>1</sup>University of Science and Technology of China
</div>

## 🔍 Overview
<div align="center">
<img width="800" alt="image" src="Figures/framework.jpg">
</div>

## 🗨 Description

Few-shot segmentation (FSS) aims to segment objects of new categories given only a handful of annotated samples. Previous works focus their efforts on exploring the support information while paying less attention to the mining of the critical query branch. In this paper, we rethink the importance of support information and propose a new query-centric FSS model Adversarial Mining Transformer (AMFormer), which achieves accurate query image segmentation with only rough support guidance or even weak support labels. The proposed AMFormer enjoys several merits. First, we design an object mining transformer (G) that can achieve the expansion of incomplete region activated by support clue, and a detail mining transformer (D) to discriminate the detailed local difference between the expanded mask and the ground truth. Second, we propose to train G and D via an adversarial process, where G is optimized to generate more accurate masks approaching ground truth to fool D. We conduct extensive experiments on commonly used Pascal-5i and COCO-20i benchmarks and achieve state-of-the-art results across all settings. In addition, the decent performance with weak support labels in our query-centric paradigm may inspire the development of more general FSS models.


## 🖊 TODO
- [ ] Release code (coming soon)
- [ ] Release models
- [ ] Demo
