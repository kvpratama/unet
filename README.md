# Various U-Net Model in Pytorch


### U-Net
The original U-Net model

**U-Net: Convolutional Networks for Biomedical Image Segmentation**

https://arxiv.org/abs/1505.04597 

### U-Net Attention
**Attention U-Net: Learning Where to Look for the Pancreas**

https://arxiv.org/abs/1505.04597 

### RU-Net and R2U-Net
**Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation**

https://arxiv.org/abs/1802.06955


## Run The Training and Evaluation
``python train.py --checkpoint_dir "./saved_models/unet_carvana" --model_name "unet"``

Current available ``model_name`` is ``unet``, ``unet_attention``, ``unet_recurrent``, ``unet_r2``

To evaluate the performance run ``python test.py --checkpoint_dir "./saved_models/unet_carvana"``


## Results
#### Carvana Image Masking Challenge
This dataset can be downloaded from https://www.kaggle.com/c/carvana-image-masking-challenge

It contains 5087 images that I split into 4028 training data and 1059 test data

|Model   	        |Accuracy 	    |Sensitivity  	|Specificity  	|Precision  	|F1  	        |Jaccard  	    |Dice
|---	            |---	        |---	        |---	        |---	        |---	        |---	        |---
|U-Net  	        |0.998279817    |0.995580243    |**0.998927329**|**0.99626272** |**0.995910721**|**0.991872912**|**0.995911221**    
|U-Net Attention  	|**0.998282434**|**0.996596236**|0.9986885      |0.99514082     |0.995856454    |0.991767802    |0.995856954  	    
|RU-Net  	        |0.993043424    |0.988357045    |0.993877302    |0.979795889    |0.98386901     |0.968556633    |0.98386951
|R2U-Net  	        |0.98894883     |0.966178541    |0.99462        |0.981679731    |0.972566577    |0.948562334    |0.972567077