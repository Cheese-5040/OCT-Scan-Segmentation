# OCT-Scan-Segmentation
This is a small project done to learn how to apply image processing on Optical Coherence Tomography Images 

## Abstract
Due to current advancements in technology, the applications of computer vision had expanded to the domain of biomedical science, and it seeks to automate tasks that the human visual system does with higher precision and accuracy. In this research, Semantic Segmentation is applied to locate and highlight different layers of Optical Coherence Tomography scans (OCT).  For this task, I utilized a fully convolutional network model for the task, which is U-NET architecture to localize areas of OCT images. The rationale behind using this architecture is due to the ability of U-NET to localize and distinguish layers through segmentation of every pixel in the input image and funnel them to their category of classes.  In this case, 6 classes were used as the model segments the 6 most obvious layers in OCT scans. Through the use of Annotated Retinal OCT images database (AROI Database) from Sestre Milosrdnice University Hospital Center, I have a total of 1105 OCT scans with both raw and masked as the initial dataset. After training the U-NET model with 50 epochs, the accuracy for train data is 0.98 with loss of 0.03, while testing on new images shown that the dice coefficient is 0.9 and the mean intersection over union score is 0.87. This shows that it considerably performs well over untested data. However, it is important to note that the training set had very similar images and masks as they are all scanned from 24 patients, and resource limitation towards the deep learning model. With this in mind, I am interested to further develop segmentation and localization of OCT images such that abnormalities (like Diabetic Macular Edema) can be detected with high accuracy and treatment can be provided. 

## Background
This program is coded in Google Collaboratory as it can utilize the free Tesla K80 GPU since it is more powerful than my current laptop and the source code is linked at the bottom of this report. The language used to code this project is python3, with strong emphasis of Keras and Tensorflow python modules to build the model.  
Before describing the method, it is important to understand operations of a convolutional neural network (CNN) since it is the building blocks of U-NET. Firstly, CNN uses a filter which is connected to a small region of the layer before it, and it contains three hyperparameters: depth, stride and zero-padding. The depth corresponds to the amount of filters used to learn, whereas stride is how much the filter moves per pixel at a time, and padding helps regulate the spatial size of output volumes. For this application, all strides are 2 and padding is same to ensure zeros are evenly padded on the output so it has the same dimensions as the input. The function for output volume is as follows: 

Furthermore, dropout and pooling layers are commonly used between successive Conv layers in U-NET architecture to progressively reduce the special size of the representation to reduce the amount of parameters and computation in the network, hence controlling overfitting. In this case, dropout of 10-20% and max pool of 2 x 2 filters and stride 2 are used (output = 25% of input volume).  This implies that only important features of max pixels from each region of 2x2 are used, and down samples the size, resulting in easier context detection in next layer.

However, with the increase in abilities to detect features, the volumes of input size decreases drastically and there is a need for up-sampling to highlight or label the segments identified. This is a tradeoff between the network understanding WHAT the features are, but losing WHERE it is present. Therefore, we need to convert the low volume (resolution) image back to high resolution to recover and identify WHERE information are. For this application, transposed convolution is used to up-sample the parameters and expands the volume to segment the 6 classes of the image. 

## Method
The code can be broken down into three main sections, data preprocessing, deep learning, and image analysis. Firstly, for data preprocessing, the images are accessed through the docking of google drive, and fed into two arrays (T_image & T_mask), one for training images, and training mask respectively. 

## Data Preprocessing 
Scikit image package is used to import the images, and they are resized to 128, 128 per image for all 1105 images as shown in figure 3.0. Then, through prior analysis, there is only very few images with >6 layers segmented (<30 images), therefore, the mask images are cleaned to have class 0 -5 to reduce overfitting. This is done through a big loop iterating through all the images, and np.unique further shows that there are classes 0-5 left in the labelled masks. The input images are as displayed in figure 4.1 below.

After that, the masks is parsed through LabelEncoder to ensure that only [0,1,2,3,4,5] are used in T_masks and no number is skipped while T_image is normalized. Then, the inputs are split twice to be fed to the model, one with 0.1 test size (X_test, y_test) and 0.9 train size (X1, y1), and within the 0.9 training data, split into 0.2 test (X_do_not_use, y_do_not_use)  and 0.8 train (X_train, y_train). Shapes of the data are shown in fig 4.2

Categorization is applied to y_train and y_test to obtain the shpe of (795, 128, 128, 6) and (111,128, 128,6) to categorize the 6 layers. Lastly, the initial class weights are adjusted through sklearn to prevent skewing due to high frequency of common classes. Now the data preparation is complete to initiate the model. 

## Deep Learning
Fig 5.1 illustrates the UNET model that is used to train image segmentation. The left side represents the contraction path and right side represents the expansion path. It outputs images of (128,128,6). c1-c9 denotes output tensors of Convolutional layers, p1-p9 denotes output tensor of Max Pooling Layers, and u6-u9 denotes output tensors of Transposed Convolutional Layers.

For deep learning U-NET architecture, it is mainly split into two main parts (as shown in fig 5.1), which is the encoding path and decoding path as mentioned in the background. First, the input tensor of shape (128,128,1) is parsed and two convolutional layers with 16 filters, 3x3 filter size and RELU activation are applied to it to only take positive values and save memory. After the first convolutional layer, 10% of the output tensor will be randomly dropped, and after the two layers, Max Pooling of 2x2 filter size is applied to achieve the shape of (64,64,16) at p1. This parsing to two conv layers is repeated for 32, 64, 128, and 256 filters with the same hyperparameters respectively, until it reaches the fifth convolutional layer, c5. At this point, the data is reduced to the shape of (8,8,256), and it reaches the middle point of the model where encoding stops and decoding starts and recovers localization through up-sampling. For improvement in precision, every process of decoding will skip connections through concatenation of output of convolution layer with feature maps of the encoder at the same rank: 
  
	u6 = u6 + c4,  u7 = u7 + c3,  u8 = u8 + c2,  u9 = u9 + c1
  
The decoding process of one level are as follows: input tensor -> transposed convolutional layer -> concatenation -> convolution -> dropout -> convolution -> output tensor. Two consecutive regular convolution layers are applied after every concatenation for a more precise output, and it is repeated four times until the output tensor is of shape (128,128,6).  The final convolution layer will have 6 filters of 1x1 and SoftMax activation function to categorize the 6 classes in the initial input image. This encoding of 4 blocks and decoding of 4 blocks gives the architecture a symmetric U-shape, and the relationship of shapes for the whole process are as follows: 

  input (128, 128, 1) -> encoding through convolution, maxpool and dropout -> (8,8,256) -> decoding through transposed convolution, convolution and dropout -> output (128,128,6) 

Lastly, adam optimizer is used for stochastic gradient descent since it combines both RMSprop and momentum, with high efficiency on noisy data of large params, and helps achieve global minimum effectively.  After fitting the model through 50 epochs, the results of the model are as shown in Figure 5.2, and Figure 5.3. The model is then saved in drive to be loaded for future test and use. 

## Image Analysis
After training the U-NET model, unseen data is fed to generate segmentation of layers for analysis of the architecture. This is because high pixel accuracy in trained dataset does not always imply superior segmentation ability. The results are analysed in two ways, intersection over union (IoU) and dice coefficient. 

IoU is the area of overlap between predicted segmentation and ground truth divided by area of union between both pictures, and the average IoU for this multi class segmentation are as shown in figure 6.1, which is 0.878. This implies that U-NET architecture is relatively good in segmentating OCT scans into 6 layers, but further investigation of those classes is needed.

With further examination in figure 6.1, the IoU for most layers are high (>90), except for class 4 with 0.58. This might be due to the thinness of this layer in both the images and masks, making it challenging to segment them with high accuracy. Unsurprisingly, the first class is having the highest IoU as it is easier to segment with class 1 taking 40% of the images. Class 6 is relatively lower as well (0.83), but it might be due to the lack of masks with class 6 segmented.

Similarly, Dice coefficient perform the same as it calculates 2*overlap area / all total area combined. Figure 6.2 shows the scores per class and the average dice score, which is 0.9. As mentioned earlier, class 4 had the least score due to it being very thin and hard to segment. However, overall the predictions are considerably similar to the naked eye, and might even be more accurate given better initial data for training. 

## Results
The results of model prediction are as shown in figures 7.1, 7.2, and 7.3, and the prediction of test image results shown that layer segmentation is effective through U-Net architecture at high similarity as masks.

## Conclusion 
Through this research experience, the program helps localization and segmentation of layers of the eye in OCT scans well through the application of U-NET architecture. The structure of it consists of a encoding / compressing part where stacks of convolution -> dropout -> convolution -> maxPool is used, and a decoding / upsampling part where transposed convolution -> concatenate -> convolution -> dropout -> convolution is used.  From the initial dataset used to train this model, it is decent in providing predictions as shown in results above. 

However, the results might be unexpected such as figure 7.3 and an error in prediction might lead to detrimental results especially in the biomedical field. Secondly, choosing to overlook higher classes (layer 7 and 8) and grouping them together for preprocessing due to the lack of data (24 patients) is not a good practice as the model will fail to segment layers more than 6. More data is required in order to make the model more reliable, and possibly greater variations in OCT scans as well. 
In the near future, I aspire to learn more regarding deep learning applications towards computer vision such that a better understanding of deep learning will result in possibly another new application of abnormally detection in Biomedical images. 

## Dataset
Dataset Source - Annotated Retinal OCT images database (AROI Database) from Sestre milosrdnice University Hospital Center: https://ipg.fer.hr/ipg/resources/oct_image_database

## References
1.	Andrej Karpathy, et al, CS231n, (2021), Convolutional Neural Networks for Visual Recognition, Stanford University, course notes, retrieved from: https://cs231n.github.io/convolutional-networks/
2.	Github: Python for microscopists, (2021), retrieved from : https://github.com/bnsreenu/python_for_microscopists
3.	Harshal. L, (17 Feb 2019), Understanding Sematic Segmentation with UNET, A Salt Identification Case Study, Towards Data Science, retrieved from : https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
4.	Kingma, D and Ba, J . (2015) Adam: A method for Stochastic Optimization. retrieved from: https://arxiv.org/pdf/1412.6980.pdf
5.	Naoki, (13 Nov 2017), Up-sampling with Transposed Convolution, retrieved from: https://naokishibuya.medium.com/up-sampling-with-transposed-convolution-9ae4f2df52d0
6.	Sohini. R, (11 Mar 2021), A Machine Learning Engineer’s Tutorial to Transfer Learning for Multi-Class Image Segmentation using U-Net, A debugging guide for image segmentation models, Towards Data Science, retrieved from: https://towardsdatascience.com/a-machine-learning-engineers-tutorial-to-transfer-learning-for-multi-class-image-segmentation-b34818caec6b




