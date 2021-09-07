# CNN ESM Emulator

Official code repository for the paper: [Toward efficient calibration of higher-resolution Earth System Models](https://www.climatechange.ai/papers/icml2021/51), 
presented at ICML 2021 Climate Change AI workshop.

Developed using Tensorflow 2.4, which is compatible with Python 3.6-3.8, CUDA 11.0, and cudnn 8.0.

## Setup
1. [Install Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a new conda environment with Python 3.8: ```$ conda create -n cnn-esm python==3.8```. 
3. Activate the environment: ```$ conda activate cnn-esm```
4. Clone this repo: ```$ git clone https://github.com/Fletcher-Climate-Group/cnn-esm-emulator.git```
5. Install the dependencies: ```$ cd cnn-esm-emulator && pip install -r requirements.txt```
6. Optionally follow steps to setup [TensorFlow for GPU](https://www.tensorflow.org/install/gpu)
7. Install the cartopy package via conda for plotting: ```$ conda install -c conda-forge cartopy```
8. Download and extract the preprocessed ESM data:  ```$ python download_data.py```

## Training (high-resolution)
Run ```$ train.py``` to train the model below on the high-resolution (f09) data using the default settings. 
The experiment will be saved under ```experiments/single-res```.

![alt_txt](resources/arch.png)

Sample test predictions from the trained model are shown below:

| Output  | Prediction | Ground-Truth |
| --- | --- | --- |
| AOD  | ![alt_txt](resources/sample_plots/sample26_AOD.png) | ![alt_txt](resources/sample_plots/sample26_AOD_gt.png) |
| CLDL | ![alt_txt](resources/sample_plots/sample26_CLDL.png) | ![alt_txt](resources/sample_plots/sample26_CLDL_gt.png) |
| FNET | ![alt_txt](resources/sample_plots/sample26_FNET.png) | ![alt_txt](resources/sample_plots/sample26_FNET_gt.png) |
| LWCF | ![alt_txt](resources/sample_plots/sample26_LWCF.png) | ![alt_txt](resources/sample_plots/sample26_LWCF_gt.png) |
| PRECT | ![alt_txt](resources/sample_plots/sample26_PRECT.png) | ![alt_txt](resources/sample_plots/sample26_PRECT_gt.png) |
| QRL | ![alt_txt](resources/sample_plots/sample26_QRL.png) | ![alt_txt](resources/sample_plots/sample26_QRL_gt.png) |
| SWCF | ![alt_txt](resources/sample_plots/sample26_SWCF.png) | ![alt_txt](resources/sample_plots/sample26_SWCF_gt.png) |

Code for multi-resolution training to be uploaded soon...

