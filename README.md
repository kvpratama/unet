# Various U-Net Model in Pytorch


### U-Net
The original U-Net model

**U-Net: Convolutional Networks for Biomedical Image Segmentation**

https://arxiv.org/abs/1505.04597 
<p align="left">
  <img src="https://github.com/kvpratama/unet/blob/master/assets/unet.png" width="500">
</p>

### U-Net Attention
**Attention U-Net: Learning Where to Look for the Pancreas**

https://arxiv.org/abs/1804.03999
<p align="left">
  <img src="https://github.com/kvpratama/unet/blob/master/assets/unet_attention.png" width="500">
</p>

### RU-Net and R2U-Net
**Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation**

https://arxiv.org/abs/1802.06955
<p align="left">
  <img src="https://github.com/kvpratama/unet/blob/master/assets/unet_recurrent.png" width="800">
</p>

### UNet++
**UNet++: A Nested U-Net Architecture for Medical Image Segmentation**

https://arxiv.org/pdf/1807.10165
<p align="left">
  <img src="https://github.com/kvpratama/unet/blob/master/assets/unet_nested.png" width="500">
</p>


## Run The Training and Evaluation
``python train.py --checkpoint_dir "./saved_models/unet_carvana" --model_name "unet" --dataset_name "carvana"``

Currently available ``model_name`` is ``unet``, ``unet_attention``, ``unet_recurrent``, ``unet_r2``

Currently available ``dataset_name`` is ``carvana``, ``isic``

To evaluate the performance run ``python test.py --checkpoint_dir "./saved_models/unet_carvana"``


## Results
#### Carvana Image Masking Challenge
This dataset can be downloaded from https://www.kaggle.com/c/carvana-image-masking-challenge

It contains 5087 images that I split into 4028 training data and 1059 test data

To train with this dataset, put it in the following structure
```
data
└── carvana
    ├── train
    └── train_mask
```

|Model   	        |Accuracy 	    |Sensitivity  	|Specificity  	|Precision  	|F1  	        |Jaccard  	    |Dice
|---	            |---	        |---	        |---	        |---	        |---	        |---	        |---
|U-Net  	        |0.998279817    |0.995580243    |**0.998927329**|**0.99626272** |**0.995910721**|**0.991872912**|**0.995911221**    
|U-Net Attention  	|**0.998282434**|**0.996596236**|0.9986885      |0.99514082     |0.995856454    |0.991767802    |0.995856954  	    
|RU-Net  	        |0.993043424    |0.988357045    |0.993877302    |0.979795889    |0.98386901     |0.968556633    |0.98386951
|R2U-Net  	        |0.98894883     |0.966178541    |0.99462        |0.981679731    |0.972566577    |0.948562334    |0.972567077
|UNet++  	        |0.989049781    |0.986223035    |0.989697149    |0.96720814     |0.975540318    |0.955095928    |0.975540817


#### ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection
This dataset can be downloaded from https://challenge2018.isic-archive.com/task1/

To train with this dataset, put it in the following structure
```
data
└── isic
    ├── input
    └── gt
```

|Model   	        |Accuracy 	    |Sensitivity  	|Specificity  	|Precision  	|F1  	        |Jaccard  	    |Dice
|---	            |---	        |---	        |---	        |---	        |---	        |---	        |---
|U-Net  	        |0.864781232    |0.77527608     |0.928438174    |**0.82717248** |0.723329646    |0.61480043     |0.723330083    
|U-Net Attention  	|0.873962586    |**0.800954378**|0.92850901     |0.815270585    |**0.737713671**|0.629463975    |**0.737714114**  	    
|RU-Net  	        |**0.879459107**|0.79983796     |0.935030489    |0.797679773    |0.735016992    |**0.629916732**|0.735017438
|R2U-Net  	        |0.870233406    |0.76183223     |0.936396868    |0.824444721    |0.714812878    |0.611360368    |0.714813307
|UNet++  	        |0.844096655    |0.675198708    |**0.940053562**|0.814203544    |0.645812596    |0.546377165    |0.64581299
