# Fit and Prune: Fast and Training-free Visual Token Pruning for Multi-modal Large Language Models  
[[paper](https://arxiv.org/abs/2409.10197)]

<img src="figure/main.png" alt="main_image" width="95%">

## TL;DR
We introduce **FitPrune**, a method that generates an efficient token pruning strategy for multi-modal large language models (MLLMs) by removing redundant visual tokens. FitPrune is easy to deploy and designed to meet a predefined computational budget while maintaining model performance.

## News
- **[2024/09/16]** Inference acceleration code for **LLaVA 1.5** is now released!

## TODOs

We will release the code and data in the following stages:

- [x] Release inference acceleration code for **LLaVA 1.5**.
- [ ] Release inference acceleration code for **LLaVA Next** and **LLaVA-HR**.
- [ ] Release statistical analysis scripts.

## Demos
Here are some example results showing the pruning efficiency with different compression rates on LLaVA1.5:

<p align="center">
  <div style="display: inline-block; text-align: center; margin-right: 10px;">
    <img src="https://github.com/ywh187/FitPrune/blob/main/figure/example.gif" alt="stitch" width="100%">
  </div>
</p>


## Usage(LLaVA1.5)

### Environment Setup

```bash
git  clone -b v1.1.3 https://github.com/haotian-liu/LLaVA.git  FitPrune_LLaVA1.5
cd FitPrune_LLaVA1.5

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

### Run Inference with FitPrune on LLaVA 1.5

```bash
# example
# adjust the reduction_ratio parameter to control the token pruning rate
bash scripts/v1_5/eval/textvqa.sh  
```



## Citation

If you find **FitPrune** useful, please kindly cite our paper:

```bibtex
@article{fitprune2024,
        title={Fit and Prune: Fast and Training-free Visual Token Pruning for Multi-modal Large Language Models},
        author={Weihao Ye, Qiong Wu, Wenhao Lin, Yiyi Zhou},
        journal={arXiv preprint arXiv:2409.10197},
        year={2024}
}
```
