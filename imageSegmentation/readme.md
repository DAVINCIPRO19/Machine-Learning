#### Project about crop the major object from image
And it seems like semantic segmentation problem.
For this I use UNET neural network architecture
![UNET_architecture](https://user-images.githubusercontent.com/83222450/167202053-fe5d943c-52f0-4d22-bc48-c24d5453adb6.png)

#### Dataset:
Data set I got from https://www.kaggle.com/c/carvana-image-masking-challenge. This dataset has original image and mask of main object.

#### Model:
Architecture has shape like U. Left part is we can call down layers, right call Ups and middle part bottleneck.
As you can see in both ups and downs model layers we have by two conv2 layers. So to simplify implementation we create DoubleConv nn. Actually it consist of two conv layers.

#### Training and Result:
Model trained for (240, 190) sized image first for 3 epoch with batch size = 16 and lr = 1e-4 and accuracy was about 98% and dice score 98%. After I trained for (959, 640) and got accuracy 99.31%, dice score 98.6%

So in the result we have not very but still good cropped car.
![9c11cb43-ea53-4b31-bbdc-31095eed2f63](https://user-images.githubusercontent.com/83222450/167203781-10fe01f2-ea0e-49c0-a3a4-3c1fa3ab0617.png)

#### Conclusion:
As we saw model has pretty good result. And also it can work for another object like animal, human, etc. But for this we need dataset with masks.

