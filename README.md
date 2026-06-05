<div align="center">

# Suppressing Non-Semantic Noise in Masked Image Modeling Representations

**[Martine Hjelkrem-Tan](https://www.mn.uio.no/ifi/english/people/aca/matan/), [Marius Aasan](https://www.mn.uio.no/ifi/english/people/aca/mariuaas/),  [Rwiddhi Chakraborty](https://rwchakra.com), [Gabriel Y. Arteaga](https://gabriel-arteaga.github.io), [Changkyu Choi](https://www.linkedin.com/in/changkyu-choi-dk78042/), [Adín Ramírez Rivera](https://www.mn.uio.no/ifi/english/people/aca/adinr/)** <br>


**[DSB @ IFI @ UiO](https://www.mn.uio.no/ifi/english/research/groups/dsb/)** <br>

[![Website](https://img.shields.io/badge/Website-green)](https://dsb-ifi.github.io/soap/)
[![PaperArxiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2604.00172) <br>

![Figure 1](/assets/images/pos_noise_compact.png#gh-light-mode-only "Examples of positional noise in principal components (PC) of MIM models")
![Figure 1](/assets/images/pos_noise_compact.png#gh-dark-mode-only "Examples of positional noise in principal components (PC) of MIM models")

</div>


This repo contains code and weights for **Suppressing Non-Semantic Noise in Masked Image Modeling Representations**, accepted for CVPR 2026.

For an introduction to our work, visit the [project webpage](https://dsb-ifi.github.io/soap/). 

## Installation

The package can be installed via:

```bash
# HTTPS
pip install git+https://github.com/dsb-ifi/soap.git

# SSH
pip install git+ssh://git@github.com/dsb-ifi/soap.git
```

## Measuring Semantic Invariance and Using SOAP

We provide a [Jupyter notebook](./notebook.ipynb) that illustrates how to get started with calculating the SI score and SOAP, and gives an example usecase for salient segmentation.

## Citation

If you find our work useful, please consider citing our paper.

```
@inproceedings{hjelkremtan2026soap,
  title={Suppressing Non-Semantic Noise in Masked Image Modeling Representations},
  author={Hjelkrem-Tan, Martine and Aasan, Marius and Chakraborty, Rwiddhi and Arteaga, Gabriel Y. and Choi, Changkyu and Ram\'irez Rivera, Ad\'in},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
