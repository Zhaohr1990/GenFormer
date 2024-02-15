# GenFormer

GenFormer: A Deep-Learning-Based Approach for Generating Multivariate Stochastic Processes

Stochastic generators are essential to produce synthetic realizations that preserve target statistical properties. We propose GenFormer, a stochastic generator for spatio-temporal multivariate stochastic processes, enlighted by Transformer-based deep learning models used for time series forecasting. 

Our numerical examples involving numerous spatial locations and simulation over a long time horizon demonstrate that synthetic realizations produced by GenFormer can be reliably utilized in downstream applications due to the superior performance of deep learning models for complex and high-dimensional tasks. 

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all the data from [Google Drive](https://drive.google.com/drive/u/0/folders/1JLjhje3j-RJ4gNKtbyEE4afWR5S32idH for SDE example and https://drive.google.com/drive/u/0/folders/1Sj3Jy-Xx8jgsNOaPjM0GdP5dMWPsj5Sb for wind example). **All the datasets are well pre-processed** and can be used easily.
3. Train the model and run the simulations. We provide the notebooks for two experiments under the folder `./notebooks`. 

## Citation

If you find this repo useful, please cite our paper. 

```
@article{genformer_paper,
    author = "Zhao, H. and Uy, W.I.T.",
    title = "GenFormer: A Deep-Learning-Based Approach for Generating Multivariate Stochastic Processes",
    journal = "	arXiv:2402.02010",
    year = 2024
}
```

## Contact

If you have any questions or want to use the code, please contact zhaohr1990@gmail.com

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020



