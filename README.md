# AIMD. AI for microscopy denoising
This repository is demonstrating the use of open-source microscopy data for deep learning based image denoising and transfer learning as showcased in:
Lohr D, Meyer L, Woelk L-M, Kovacevic D, Diercks B-P, Werner R. (2025) Deep Learning-Based Image Restoration and Super-Resolution for Fluorescence Microscopy: Overview and Resources. In: Diercks B-P (ed) T-Cell Activation: Methods and Protocols. Springer US, New York, NY, p XXX–XXX

All code is available as jupyter notebooks which can be customized for application to new data.
Models are trained using one or both of the following two datasets:
1) "Fluorescence Microscopy Denoising (FMD) dataset" - CC BY-SA 4.0 license
   - https://curate.nd.edu/articles/dataset/Fluorescence_Microscopy_Denoising_FMD_dataset/24744648
   - 8 bit image data, filetype: png
   - referred to as FMD data 
2) "Fluorescence Microscopy Datasets for Training Deep Neural Networks" - CC0 license
    - http://gigadb.org/dataset/view/id/100888
    - 16 bit image data, filetype: tif
    - referred to as Hagen data
       
Details regarding samples and data acquistion can be found in their respective publications. 

If you are using this repository in your research, please cite:
Lohr D, Meyer L, Woelk L-M, Kovacevic D, Werner R. (2025) Deep Learning-Based Image Restoration and Super-Resolution for Fluorescence Microscopy: Overview and Resources. In: Diercks B-P (ed) T-Cell Activation: Methods and Protocols. Springer US, New York, NY, p XXX–XXX

# Prior to using this repository
1) clone the repository
2) create a virtual environment (conda/miniconda) and activate it
3) install pytorch following the instructions here: https://pytorch.org/get-started/locally/
4) install fastai (2.7.15) following the instructions here: https://docs.fast.ai 
5) install required dependencies using the requirement.txt 

# Loss function: Perception loss
The full loss is calculated as the sum of:
- L1 loss of prediction and ground truth
- L1 loss of featuremaps derived from 3 layers of a VGG16_bn
- L1 loss of gram matrices derived from 3 layers of a VGG16_bn

# Quality metrics  
MSE and PSNR are tracked during training. Respective metrics could be included into the loss by adjusting the notebook Perception_loss.ipynb

# Model performance on the FMD test set
## Visual assessment
![FMD](https://github.com/user-attachments/assets/9f9b50ae-2722-4779-bb43-04719c561a0a)

## Quantitative assessment
Values are PSNR / SSIM, result of FMD_analyze.ipynb
|Averages |1       |2       |4       |8       |16       |
|-------  |--------|--------|--------|--------|---------|
|Method   |        |        |        |        |         |
|Raw      |27.2 / 0.55|30.1 / 0.68|32.9 / 0.80|36.0 / 0.89|39.7 / 0.95|
|Denoised |35.1 / 0.91|36.8 / 0.93|38.1 / 0.94|39.6 / 0.96|41.4 / 0.97|

# Model performance on the Hagen test set
- Model 1: Basemodel_Hagen, trained using solely Hagen Actin training images of noise level 1
- Model 2: FMD_to_Hagen_Transfer, pre-trained using FMD data and re-trained using the same Actin images
- Model 3: FMD_to_Hagen_all_Transfer, pre-trained using FMD training data and re-trained using all Hagen training images

## Visual assessment
- Model 1 performs well for the Actin test data, but struggles with higher noise levels and other samples. 
- Model 2 still struggles with the higher noise levels, but can cope with other samples. 
- Model 3 performs well on all noise levels and all samples.

![Hagen](https://github.com/user-attachments/assets/231a9301-e9ef-4341-a477-08245b58c695)

## Quantitative assessment of Actin and Membrane images
Values are PSNR / SSIM, result of Hagen_analyze.ipynb
|Sample  |Actin 20x| Actin 60x noise 1| Actin 60x noise 2|Actin confocal|Membrane|
|---|---|---|---|---|---|
|Method||||||
|Raw|24.2 / 0.34|28.0 / 0.53|18.4 / 0.17|24.7 / 0.39|29.5 / 0.61|
|1)|31.0 / 0.34|37.6 / 0.53|26.0 / 0.17|26.8 / 0.66|26.4 / 0.70|
|2)|31.2 / 0.34|37.2 / 0.53|25.5 / 0.17|26.8 / 0.64|30.9 / 0.78|
|3)|32.0 / 0.34|37.1 / 0.53|28.5 / 0.80|28.4 / 0.82|35.2 / 0.92|

## Quantitative assessment of Mito and Nucleus images
Values are PSNR / SSIM
|Sample  |Mito 20x| Mito 60x noise 1| Mito 60x noise 2|Mito confocal|Nucleus|
|---|---|---|---|---|---|
|Method||||||
|Raw|24.6 / 0.35|28.0 / 0.51|20.1 / 0.30|22.2 / 0.37|24.7 / 0.39|
|1)|31.0 / 0.87|35.4 / 0.92|25.4 / 0.46|26.4 / 0.59|32.9 / 0.82|
|2)|31.2 / 0.87|36.3 / 0.93|25.1 / 0.43|26.0 / 0.56|32.9 / 0.82|
|3)|32.7 / 0.92|37.8 / 0.95|27.8 / 0.84|28.4 / 0.69|34.7 / 0.87|

# Repository application
Training and inference notebooks are commented to ease application. Information below are meant to provide additional guidance!

## Training models and reproducing results
1) download the open-source data, if you want to re-train models or reproduce results using notebooks of this repository
2) place the FMD file (.zip) from the FMD dataset in 'Daten/FMD_rawdata'
3) place the 16-bit TIFF files from the Hagen dataset in 'Daten/Hagen_rawdata'
5) prepare training data structure for repository (see below) by running FMD_image_to_tiles.ipynb and Hagen_image_to_tiles.ipynb
6) run notebooks Basemodel_FMD.ipynb, Basemodel_Hagen.ipynb, or FMD_to_Hagen_Transfer to train the models yourself
7) run notebooks Inference_Basemodel_FMD.ipynb, Inference_Basemodel_Hagen.ipynb, or Inference_FMD_to_Hagen.ipynb to generate predictions for Testsets
8) turn predicted patches/tiles back to images using FMD_tiles_to_images.ipynb and Hagen_tiles_to_images.ipynb
9) calculate PSNR and SSIM for the test sets using FMD_analyze.ipynb and Hagen_analyze.ipynb

## Training models with your own data
1) choose notebook depending on datatype of your images or adjust your images accordingly:
- Basemodel_FMD.ipynb if you use 8-bit data
- Basemodel_Hagen.ipynb if you use 16-bit data
- FMD_to_Hagen_Transfer.ipynb if you use 16-bit data

2) put your training data in respective folders ("Daten/YourGT" and "Daten/YourNoisy")
3) adjust variables path_GT and path_noisy in the notebook to path_GT = path/'Daten/YourGT' and path_noisy = path/'Daten/YourNoisy'
4) adjust the filename to save 'YourModelName' and avoid overwriting prior models
5) run the notebook (set a proper learning rate using "learn_den.lr_find()" and set the number of epochs to train)
6) the training progress is printed in the notebook and also saved in a .csv file

## Directly applying our models to your images
1) decide which model you want to use and run the corresponding notebook (do take note, that Hagen models use 16b input images, while the FMD model uses 8 bit input images):
- Inference_Basemodel_FMD.ipynb
- Inference_Basemodel_Hagen.ipynb
- Inference_FMD_to_Hagen.ipynb (trained with either Actin or all Hagen data)

2) adjust path_GT and path_noisy variables to your test/target data folder (path_GT = path/'Daten/YourTest' and path_noisy = path/'Daten/YourTest')
3) adjust path_test variable to your test/target data folder (path_test = path/'Daten/YourTest')
4) adjust path_preds variable as a folder to save your data in, e.g. path_preds = path/'Daten/YourPredictions'
5) run the notebook for inference  

## Directly applying models you trained 
1) run steps 1-4 as listed above
2) adjust the filename of the model to load in the line: learn_den.load('YourModelName').to_fp16()
3) run the notebook for inference
   
## Data structure of the repository
Datasets were downloaded from their respective sites and stored in the folders:
- 'Daten/FMD_rawdata'
- 'Daten/Hagen_rawdata'

Running the notebooks:
- FMD_images_to_tiles.ipynb
- Hagen_images_to_tiles.ipynb

in the folder "Daten" sorts the "rawdata" in training and test images, adhering to the following structure:

      - FMD_GT          # ground truth image data for training, result from running (n=60 000):
      - FMD_noisy       # noisy image data for training, result from running (n=60 000):
      - FMD_testmix   - avg2                # noisy test data, averaged from 2 image acquistions
                      - avg4                # noisy test data, averaged from 4 images acquistions
                      - avg8                # noisy test data, averaged from 8 images acquistions
                      - avg16               # noisy test data, averaged from 16 images acquistions                    
                      - gt                  # ground truth test data, averaged from 50 acquistions                    
                      - raw                 # noisy test data, single (1) image acquisition
                      - noisy_tiles         # noisy 256x256 image tiles used as model input
                    
      - Hagen_actin_GT          # ground truth tiles of actin images for training (n=3859)
      - FMD_actin_noisy         # noisy tiles of actin images for training (n=3859)      
      - Hagen_GT                # ground truth tiles of the complete dataset for training (n=36 222)
      - Hagen_noisy             # noisy tiles of the complete dataset for training (n=36 222)
      - Hagen_testmix - gt_images           # gt test data, averaged from 400 image acquistions
                      - noisy_images        # noisy test data averaged from 2, 4, 8, 16 
                      - noisy_tiles         # noisy 256x256 image tiles used as model input
                    
Pre- and post-processing notebooks:
- FMD_image_to_tiles.ipynb: generates 256x256 pixel tiles for FMD testset for inference 
- FMD_tiles_to_images.ipynb: generates 512x512 images (original size) from the tiles after inference
- FMD_analyze.ipynb: provides PSNR and SSIM for the FMD testset 
- Hagen_image_to_tiles.ipynb: generates 256x256 pixel tiles for Hagen training and test images  
- Hagen_tiles_to_images.ipynb: generates images of original size from tiles (test set) after Inference
- Hagen_analyze.ipynb: provides PSNR and SSIM for the Hagen testset

```
@incollection{Lohr2025,
    author       = {Lohr, David and Meyer, Lina and Woelk, Lena-Marie and Kovacevic, Dejan and Diercks, Björn-Philipp and Werner, René},
    title        = {Deep Learning-Based Image Restoration and Super-Resolution for Fluorescence Microscopy: Overview and Resources},
    booktitle    = {T-Cell Activation: Methods and Protocols},
    editor       = {Diercks, Björn-Philipp},
    year         = {2025},
    publisher    = {Springer US},
    address      = {New York, NY},
    pages        = {XXX--XXX}
}
```
