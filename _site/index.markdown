<!-- ---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
--- -->
<div align="center">

# Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models

### [Manli Shu]()<sup>1</sup>, [Chaowei Xiao](https://xiaocw11.github.io/)<sup>2</sup>, [Weili Nie](https://weilinie.github.io/)<sup>2</sup>, [De-An Huang](https://ai.stanford.edu/~dahuang/)<sup>2</sup>, [Zhiding Yu](https://chrisding.github.io/)<sup>2</sup>,      
### [Tom Goldstein](https://www.cs.umd.edu/~tomg/)<sup>1</sup>, [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/)<sup>2,3</sup>
### <sup>1</sup> University of Maryland, <sup>2</sup> NVIDIA, <sup>3</sup> Caltech

### [<ins>arxiv</ins>]() &nbsp; [<ins>code</ins>]()
</div>
<br>

***Abstract***: Pre-trained vision-language models (e.g., CLIP) have shown impressive zero-shot
generalization in various downstream tasks with properly designed text prompts. Instead of relying on hand-engineered prompts, recent works learn prompts using training data from downstream tasks, but this can be expensive and hard to generalize to new tasks and distributions. To this end, **we propose test-time prompt tuning (*TPT*)** as the first prompt tuning method that can **learn adaptive prompts on the fly with a single test sample**. TPT optimizes the prompt by minimizing the entropy with confidence selection so that the model has consistent predictions across different augmented views of each test sample. In the setting of evaluating natural distribution shifts, TPT improves the zero-shot top-1 accuracy of CLIP by 3.6% on average, even surpassing previous prompt tuning approaches that additionally require task-specific training data. In the setting of evaluating across-dataset generalization with unseen categories, TPT performs on par with the state-of-the-art approach that uses training data.  
<br /> 

## Test-time Prompt Tuning
TPT tunes adaptive prompts on the fly with a single test sample, **without the need of training data or annotations**. TPT optimizes the prompt to encourage consistent predictions across augmented views of the same test image by minimizing the marginal entropy. In addition, we introduce ***confidence selection*** to filter out noisy augmentations.

<p align = "center">
<img src = "https://github.com/azshue/TPT/blob/gh-pages/assets/tpt-intro.png?raw=true">
</p>
<p align = "center">
An overview of Test-time Prompt Tuning (TPT)
</p>
<br />

We summarize existing prompt tuning methods for CLIP, and compare the differences between TPT and existing methods. 
<!-- We focus on three preferred properties of a prompting strategy, and use them to categorize the methods. "Learnable" means the prompt is optimized based on certain objective functions. "No training data" means that no additional data are needed for tuning the prompt. "Input-adaptive" means the prompt can be adaptive to each input instance. -->

<div align="center">

| Prompt Type  | Learnable | No training data | Input-adaptive |
|--------------|:---------:|:----------------:|:--------------:|
| [Hand-crafted](https://arxiv.org/abs/2103.00020) |           |       &#10003;    |                |
| [CoOp](https://arxiv.org/abs/2109.01134)          |  &#10003;  |                  |                |
| [CoCoOp](https://arxiv.org/abs/2203.05557)       |  &#10003;  |                  |   &#10003;      |
| TPT (ours)    |  &#10003;  |       &#10003;    |    &#10003;     |

</div>
<br />

## Evaluation


### Generalization to Natural Distribution Shifts

<!-- We evaluate model's robustness to natural distribution shifts on 4 ImageNet Variants as follows, which have been considered as out-of-distribution (OOD) data for ImageNet in previous work. -->
Compared to existing prompt tuning methods that requires training data, TPT generalizes better to data distribution shifts. Note that among the methods in the table below, CoOp and CoCoOp are tuned on ImageNet using 16-shot training data per category. Baseline CLIP, prompt ensemble and TPT (ours) do not requires training data.


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

### Cross-Datasets Generalization
<!-- Pre-trained vision-language models like CLIP are ideal for ``open-world" problems. For example, we can apply CLIP to classify arbitrary categories in a zero-shot manner in image classification. However, a prompt tuned on a specific downstream dataset can be less generalizable to categories outside its training set. We conduct cross-dataset evaluation on image classification, where we consider 10 different source/target datasets.  -->

In each matrix $A$, $A_{i, j}$ is the **normalized relative improvement** on the $j_{th}$ dataset of using the prompt tuned on the $i$-th dataset. The value $A_{i, j}$ stands for **how well a method trained on a source dataset $i$ performs on a target dataset $j$**, in comparison with a zero-shot CLIP baseline (using a hand-crafted prompt). Thus, the higher, the better.
The last row is the performance of TPT, which is not tuned on any source dataset. The last column summarize the average improvement over 10 datasets, measuring the overall generalization ability across the 10 datasets.

<p align = "center">
<img src = "https://github.com/azshue/TPT/blob/gh-pages/assets/cross-datasets-figures.png?raw=true">
</p>
<p align = "center">
Cross-dataset improvement normalized by the zero-shot baseline performance.
</p>


## Citation
If you find our work useful, please consider citing:
```
@article{shu2022tpt
    title={Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models},
    author={Manli, Shu and Chaowei, Xiao and Weili, Nie and De-An, Huang and Zhiding, Yu and Tom, Goldstein and Anima, Anandkumar},
    journal={arXiv preprint arXiv: },
    year={2022}
}
```

<!-- ## Acknowledgements -->

## Contact
For any questions, please contact Manli Shu (manlis@cs.umd.edu).