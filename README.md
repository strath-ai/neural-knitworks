# 🪢 Neural Knitworks
This repository contains code for the paper **"Neural Knitworks: Patched Neural Implicit Representation Networks"** published in **Pattern Recognition**:

```
@article{knitworks,
  title = {Neural Knitworks: Patched neural implicit representation networks},
  journal = {Pattern Recognition},
  volume = {151},
  pages = {110378},
  year = {2024},
  issn = {0031-3203},
  doi = {https://doi.org/10.1016/j.patcog.2024.110378},
  url = {https://www.sciencedirect.com/science/article/pii/S0031320324001298},
  author = {Mikolaj Czerkawski and Javier Cardona and Robert Atkinson and Craig Michie and Ivan Andonovic and Carmine Clemente and Christos Tachtatzis},
}
```

The code contains the implementation for solving tasks of:
* image inpainting
* super-resolution
* denoising

## :snake: Environment

To create a conda environment:

```setup
conda env create -f environment.yml
```
This environment was set up with CUDA Version 11.2.

## :wrench: Notebook
The applications of neural knitworks can be explored by running `01-Example-Usage.ipynb`
