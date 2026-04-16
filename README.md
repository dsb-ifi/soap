<div align="center">

# Suppressing Non-Semantic Noise in Masked Image Modeling Representations

**[Martine Hjelkrem-Tan](https://www.mn.uio.no/ifi/english/people/aca/matan/), [Marius Aasan](https://www.mn.uio.no/ifi/english/people/aca/mariuaas/),  [Rwiddhi Chakraborty](https://rwchakra.com), [Gabriel Y. Arteaga](https://gabriel-arteaga.github.io), [Changkyu Choi](https://www.linkedin.com/in/changkyu-choi-dk78042/), [Adín Ramírez Rivera](https://www.mn.uio.no/ifi/english/people/aca/adinr/)** <br>


**[DSB @ IFI @ UiO](https://www.mn.uio.no/ifi/english/research/groups/dsb/)** <br>

[![Website](https://img.shields.io/badge/Website-green)](https://dsb-ifi.github.io/soap/)
[![PaperArxiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2507.01654) <br>
<!-- [![PaperICCVW](https://img.shields.io/badge/Paper-ICCVW_2025-blue)](https://eclr-workshop.github.io/)
[![NotebookExample](https://img.shields.io/badge/Notebook-Example-orange)](https://nbviewer.jupyter.org) <br> -->

![SPoT Figure 1](/assets/images/pos_noise_compact.png#gh-light-mode-only "Examples of positional noise in principal components (PC) of MIM models")
![SPoT Figure 1](/assets/images/pos_noise_compact.png#gh-dark-mode-only "Examples of positional noise in principal components (PC) of MIM models")

</div>

## SPoT: Subpixel Placement of Tokens

This repo contains code and weights for **Suppressing Non-Semantic Noise in Masked Image Modeling Representations**, accepted for CVPR 2026.

For an introduction to our work, visit the [project webpage](https://dsb-ifi.github.io/soap/). 

## Installation

The package can currently be installed via:

```bash
# HTTPS
pip install git+https://github.com/dsb-ifi/soap.git

# SSH
pip install git+ssh://git@github.com/dsb-ifi/soap.git
```

## Measuring Semantic Invariance and Using SOAP

We provide a [Jupyter notebook](./get_started.ipynb) that illustrates ...

## Citation

If you find our work useful, please consider citing our paper.

```
@inproceedings{hjelkremtan2025soap,
  title={Suppressing Non-Semantic Noise in Masked Image Modeling Representations},
  author={Hjelkrem-Tan, Martine and Aasan, Marius and Chakraborty, Rwiddhi and Arteaga, Gabriel Y. and Choi, Changkyu and Ram\'irez Rivera, Ad\'in},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```