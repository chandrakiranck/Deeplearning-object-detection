  # product_detection_chandrakiranreddy_gudipally
We use Grocery Dataset collected by Idea Teknoloji, Istanbul, Turkey. It contains 354 tobacco shelves images collected from ~40 locations with 4 cameras.
For the sake of clarity create "data" directory with "images" directory inside. Download and unpack grocery dataset images there.
The images dataset can be downloaded using wget and unpacked using tar:

https://github.com/gulvarol/grocerydataset/releases/download/1.0/GroceryDataset_part1.tar.gz
https://github.com/gulvarol/grocerydataset/releases/download/1.0/GroceryDataset_part2.tar.gz

downloaded ShelfImages.tar.gz(contains train and test splits) and replaced with 
GroceryDataset_part1/ShelfImages with this.
https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz

After downloading and unpacking your grocery-shelves/data/images/ directory should appear as follows:
+ your grocery-shelves/data/images/
+ BrandImages
+ BrandImagesFromShelves
+ ProductImages
+ ProductImagesFromShelves
+ ShelfImages

Unfortunately, some of images are rotated. It can be fixed using following commands:
convert C1_P03_N1_S2_1.JPG -rotate 180 C1_P03_N1_S2_1.JPG
convert C1_P03_N1_S2_2.JPG -rotate 180 C1_P03_N1_S2_2.JPG
convert C1_P03_N1_S3_1.JPG -rotate 180 C1_P03_N1_S3_1.JPG
convert C1_P03_N1_S3_2.JPG -rotate 180 C1_P03_N1_S3_2.JPG
convert C1_P03_N1_S4_1.JPG -rotate 180 C1_P03_N1_S4_1.JPG
convert C1_P03_N1_S4_2.JPG -rotate 180 C1_P03_N1_S4_2.JPG
convert C1_P03_N2_S3_1.JPG -rotate 180 C1_P03_N2_S3_1.JPG
convert C1_P03_N2_S3_2.JPG -rotate 180 C1_P03_N2_S3_2.JPG
convert C1_P03_N3_S2_1.JPG -rotate 180 C1_P03_N3_S2_1.JPG
convert C1_P12_N1_S2_1.JPG -rotate 180 C1_P12_N1_S2_1.JPG
convert C1_P12_N1_S3_1.JPG -rotate 180 C1_P12_N1_S3_1.JPG
convert C1_P12_N1_S4_1.JPG -rotate 180 C1_P12_N1_S4_1.JPG
convert C1_P12_N1_S5_1.JPG -rotate 180 C1_P12_N1_S5_1.JPG
convert C1_P12_N2_S2_1.JPG -rotate 180 C1_P12_N2_S2_1.JPG
convert C1_P12_N2_S3_1.JPG -rotate 180 C1_P12_N2_S3_1.JPG
convert C1_P12_N2_S4_1.JPG -rotate 180 C1_P12_N2_S4_1.JPG
convert C1_P12_N2_S5_1.JPG -rotate 180 C1_P12_N2_S5_1.JPG
convert C1_P12_N3_S2_1.JPG -rotate 180 C1_P12_N3_S2_1.JPG
convert C1_P12_N3_S3_1.JPG -rotate 180 C1_P12_N3_S3_1.JPG
convert C1_P12_N3_S4_1.JPG -rotate 180 C1_P12_N3_S4_1.JPG
convert C1_P12_N4_S2_1.JPG -rotate 180 C1_P12_N4_S2_1.JPG
convert C1_P12_N4_S3_1.JPG -rotate 180 C1_P12_N4_S3_1.JPG
convert C3_P07_N1_S6_1.JPG -rotate -90 C3_P07_N1_S6_1.JPG

We'll use data from two folders:

### ShelfImages


Directory contains JPG files named the same way as C3_P06_N3_S3_1.JPG file:
C3_P06 - shelf id
N3_S3_1 - planogram id

Shelf images/Train for training
shelfimages/test for testing the data


### ProductImagesFromShelves

Directory contains png files grouped by category named the same way as C1_P01_N1_S2_1.JPG_1008_1552_252_376.png file:
C1_P01_N1_S2_1.JPG - shelf photo file
1008 - x
1552 - y
252 - w
376 - h

## Train/Validation/Test Split
It's a good practice to split the data into three categories: train (neural network training), validation (monitor training process in order not to over fit our nn) and test (apply final performance checks).
For our purposes we will split the data into train/validation in 70/30 ratio.


### Data Augumentation

neural networks work with input of fixed size, so we need to resize our packs images to the chosen size. The size is some kind of metaparameter and you should try different variants. Logically, the bigger size you select,the better performace you'll have. Unfortunatelly it is not true, because  of over fitting. The more parameters your neural network have, the easier it became over fitted"

#### Normalized the Data

## RESNET

*  Trained  with residual network and estimated the accuracy and loss.
*  The core idea of ResNet is introducing a so-called “identity shortcut connection”
*  we got test accuracy as 0.89 

## Running Training Proccess
Please, go to pack_detector/models/ssd_mobilenet_v1 and create two subfolders: eval and train. Now we have the pack_detector folder with the following content:

+data
train.record
eval.record

pack.pbtxt
+models
+ssd_mobilenet_v1
+train
+eval
ssd_mobilenet_v1_pack.config



Done  the following steps to run the process:
Install Tensorflow Object Detection API if you haven't done it before Installation
Go to the models/research/object_detection directory

Download and untar pretrained model with:
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
tar -xvzf ssd_mobilenet_v1_coco_2017_11_17.tar.gz

Copied  pack_detector folder to models/research/object_detection\Run train process on GPU with:
python3 train.py --logtostderr --train_dir=pack_detector/models/ssd_mobilenet_v1/train/ pipeline_config_path=pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config




Now it's time to chose the best checkpoint and import it as a frozen graph. We will use this frozen graph for inference. Use the following command line, note that you need to chose your destination folder and checkpoint number (I chose 12788):
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config \--trained_checkpoint_prefix pack_detector/models/ssd_mobilenet_v1/train/model.ckpt-12788 \
--output_directory pack_detector/models/ssd_mobilenet_v1/graph_model

### Evaluating the Test Dataset

* load frozen graph
* function that executes detection
* function to display image with bounding boxes
* function for sliding window inference
* function for non-maximum suppression
* function for do_sliding_window_inference_with_nm_suppression


## Results Saved in

Created a folder Result and created a json file named images2products.json

# Metrics for object detection
  
The motivation of this project is the lack of consensus used by different works and implementations concerning the **evaluation metrics of the object detection problem**.   


## Important definitions  

### Intersection Over Union (IOU)

Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between two bounding boxes. It requires a ground truth bounding box  and a predicted bounding box . By applying the IOU we can tell if a detection is valid (True Positive) or not (False Positive).
IOU is given by the overlapping area between the predicted bounding box and the ground truth bounding box divided by the area of union between them:  


IOU is given by the overlapping area between the predicted bounding box and the ground truth bounding box divided by the area of union between them

### True Positive, False Positive, False Negative and True Negative
Some basic concepts used by the metrics:

*  True Positive (TP): A correct detection. Detection with IOU ≥ threshold
*  False Positive (FP): A wrong detection. Detection with IOU < threshold
*  False Negative (FN): A ground truth not detected
*  True Negative (TN): Does not apply. 



It would represent a corrected misdetection. In the object detection task there are many possible bounding boxes that should not be detected within an image. Thus, TN would be all possible bounding boxes that were corrrectly not detected (so many possible boxes within an image). That's why it is not used by the metrics.

### threshold: depending on the metric, it is usually set to 50%, 75% or 95%.

### Precision
Precision is the ability of a model to identify only the relevant objects. It is the percentage of correct positive predictions and is given by:



### Recall
Recall is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:





```python

```
