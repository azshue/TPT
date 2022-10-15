# Test-Time Prompt Tuning (TPT) for zero-shot generalization in Vision-Language Models

This repository provides the official PyTorch implementation of our NeurIPS 2022 paper:    

> Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models      
> Authors: *Manli Shu, Weili Nie, De-An Huang, Tom Goldstein, Anima Anandkumar, Chaowei Xiao*   

For more details, please check out our [<ins>**project page**</ins>](https://azshue.github.io/TPT/) and [<ins>**paper**</ins>](https://arxiv.org/pdf/2209.07511.pdf). 

## Overview
This repository contains the implementation of TPT for image classification with a pre-trained CLIP. We consider 3 different initializations for test-time prompt tuning:  

* Using a hand-crafted prompt as initialization (*e.g.,* "a photo of a ___")
* Using a <ins> learned soft prompt</ins> ([CoOp](https://arxiv.org/abs/2109.01134)) as initialization.
* Using the output of a <ins>trained conditional prompt learner</ins> ([CoCoOp](https://arxiv.org/abs/2203.05557)) as initialization. 



## Prerequisites

### Hardware

This implementation is for the single-GPU configuration. 

To evaluate on ImageNet, ImageNet-V2, and ImageNet-Sketch (which has 1000 classes), you will need a GPU with more than (not including) 16GB memory. This codebase is tested on a GPU with 24GB memory.
To evaluate other datasets (with less than a few hundred classes), a GPU with 16GB memory will work fine. 

### Environment 
The code is tested on PyTorch 1.7.1. 

### Datasets 

We suggest downloading all datasets to a root directory (`${data_root}`), and renaming the directory of each dataset as suggested in `${ID_to_DIRNAME}` in `./data/datautils.py`. This would allow you to evaluate multiple datasets within the same run.     
If this is not feasible, you could evaluate different datasets separately, and change the `${data_root}` accordingly in the bash script.

For out-of-distribution generalization, we consider 5 datasets:

* [ImageNet](https://image-net.org/index.php) 
* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-R](https://github.com/hendrycks/imagenet-r)
* [ImageNet-V2](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

For cross-datasets generalization, we consider 10 datasets:
* [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
* [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
* [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* [UCF101](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing)
* [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)
* [Food101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
* [SUN397](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)
* [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
* [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip)

For cross-dataset generalization, we adopt the same train/val/test splits as CoOp. Please refer to [this page](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#how-to-install-datasets), and look for download links of `split_zhou_${dataset_name}.json`, and put the json files under `./data/data_splits/`. 


## Run TPT

We provide three bash scripts under `./scripts`. You can modify the paths and other args in the scripts.     

An example to run TPT with CoOp initialization on out-of-distribution datasets:
```
bash ./scripts/test_coop.sh I/A/V/R/K.
```

The command line arg `${testsets}` can be multiple test datasets split by "/" (, which are stored under the same root dir `${data_root}`).    
Note that for simplicity, we use `set_id` to denote different datasets. A complete list of `set_id` can be found in `${ID_to_DIRNAME}` in `./data/datautils.py`. 


### Main Results

#### Out-of-Distribution Generalization

<div align="center">

| Method           | ImageNet(IN) | IN-A | IN-V2 | IN-R | IN-Sketch | Average | OOD Average |
|------------------|:--------:|:----------:|:-----------:|:----------:|:---------------:|:-------:|:-----------:|
| [CLIP-RN50](https://arxiv.org/abs/2103.00020)       |   58.16  |    21.83   |    51.41    |    56.15   |      33.37      |  44.18  |    40.69    |
| [Ensembled prompt](https://arxiv.org/abs/2103.00020)|   59.81  |    23.24   |    52.91    |    **60.72**   |      35.48      |  46.43  |    43.09    |
| [CoOp](https://arxiv.org/abs/2109.01134)            |   <ins>63.33</ins>  |    23.06   |    55.40    |    56.60   |      34.67      |  46.61  |    42.43    |
| [CoCoOp](https://arxiv.org/abs/2203.05557)          |   62.81  |    23.32   |    <ins>55.72    |    57.74   |      34.48      |  46.81  |    42.82    |
| TPT (ours)             |   60.74  |    <ins>26.67   |     54.7    |    <ins>59.11   |      <ins>35.09      |  <ins>47.26  |    <ins>43.89    |
| TPT + CoOp       |   **64.73**  |   **30.32**   |    **57.83**    |    58.99   |      **35.86**      |  **49.55**  |    **45.75**    |
| TPT + CoCoOp     |   62.93  |    27.40   |    56.60    |    59.88   |      35.43      |  48.45  |    44.83    |

</div>
<br />

#### Cross-Dataset Generalization

In each matrix $A$, $A_{i, j}$ is the **normalized relative improvement** on the $j_{th}$ dataset of using the prompt tuned on the $i$-th dataset. The value $A_{i, j}$ stands for **how well a method trained on a source dataset $i$ performs on a target dataset $j$**, in comparison with a zero-shot CLIP baseline (using a hand-crafted prompt). Thus, the higher, the better.
The last row is the performance of TPT, which is not tuned on any source dataset. The last column summarizes the average improvement over 10 datasets, measuring the overall generalization ability across the 10 datasets.

<p align = "center">
<img src = "https://github.com/azshue/TPT/blob/gh-pages/assets/cross-datasets-figures.png?raw=true">
</p>
<p align = "center">
Cross-dataset improvement normalized by the zero-shot baseline performance.
</p>


## Citation
If you find our code useful or our work relevant, please consider citing: 
```
@inproceedings{shu2022tpt,
  author    = {Manli, Shu and Weili, Nie and De-An, Huang and Zhiding, Yu and Tom, Goldstein and Anima, Anandkumar and Chaowei, Xiao},
  title     = {Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models},
  booktitle = {NeurIPS},
  year      = {2022},
}
```

## Acknowledgements
We thank the authors of [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) for their open-source implementation and instructions on data preparation. 
