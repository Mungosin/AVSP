# AVSP
[Analysis of Massive Data Sets](https://www.fer.unizg.hr/en/course/aomds)

## Project theme: Extracting Deep Features for Image Recommendation

Project is written in Python 2 using Tensorflow framework for feature extraction. Dataset we tested on is the OpenImage dataset. Due to the time it required to download the images we stopped after downloading a little bit over 900.000 images. Features were extracted using a pretrained Inception-v4 net trained on ImageNet dataset. Feature vector is created by concatenating a max pool with a kernel of image size for each layer. That means if the output of a convolution is 12x12x1024 for each image we used a max pool layer with kernels [1, 12, 12, 1] as defined in Tensorflow. The result of the max pool for the given image is a vector of size [1, 1, 1024] which was then flattened and concatenated for every convolution layer. Resulting vector has ~16000 features which is then reduced with principal component analysis (PCA) to size of just 300 elements later used for similarity comparisons. Angular and Euclidean distance was used as a metric during vector comparisons with numpy queries. Second implementation for querying was done using a library called [NMSLIB](https://github.com/searchivarius/nmslib) which creates indexes from vectors for efficient querying. Instructions on how to setup everything can be found below.

## Table of contents

<a href="#Req">Requirements</a><br>
<a href="#Data">Getting the dataset</a><br>
<a href='#Tensorflow'>Getting Tensorflow models</a><br>
<a href='#Python'>Python packages</a><br>
<a href='#PCA'>PCA model</a><br>
<a href='#NMSLIB'>NMSLIB installation</a><br>
<a href='#Results'>Example results</a><br>

## Requirements
<a id='Req'></a>

```
Python 2
```

## Getting the dataset
<a id='Data'></a>

The code will work on any dataset, it expects to get a root folder which will contain only images and/or folders with images.

The dataset we used was the OpenImage dataset and we downloaded it using this paralelized [downloader](https://github.com/ejlb/google-open-image-download).

After u clone the repository and get the CSV files you execute this in your terminal:  
python2 download.py [CSV_PATH] [OUTPUT_FOLDER_PATH] 

## Getting Tensorflow models setup
<a id='Tensorflow'></a>
Open the terminal and change the directory to the project root folder and run this in the terminal:
```
git clone https://github.com/tensorflow/models/
```

When the repository downloads, get the required checkpoint by running:  
```
wget -O InceptionV4.tar.gz http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
```
Untar the downloaded file by executing:  
```
tar -xvf InceptionV4.tar.gz
```
  
## Python packages
<a id='Python'></a>
If you don't have an NVIDIA GPU then change the tensorflow-gpu in requirements.txt to just tensorflow before running the installation.
  
Now position yourself in the project root folder.
  
To use the code we suggest creating a virtual environment.  
First install virtual environment with:  
```
pip install virtualenv
```  
  
  
After that create your virtual environment:  
```
virtualenv -p /usr/bin/python2 env
```  
  
  
Activate the environment with:  
```
source ./env/bin/activate
```
  
  
Install the prerequisite packages:  
```
pip install -r requirements.txt
```  
  
  
After this install either ipython2 or jupyter notebook and install the required kernel so you can access the environment
You should be ready to go with just executing this line in the terminal:  
```
python -m ipykernel install --user --name=[Name you desire]
```
* Our kernel name is AVSP. If you name it differently then you will get an error saying there is no kernel named AVSP. To resolve that just choose the kernel you just created from the dropdown list and click set.
  
  
To get the progress bar to work in the notebook run this command in your terminal:  
```
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```  
  
  
Now you're ready to run the notebook, in the terminal enter:  
```
ipython2 notebook
```


## PCA model
<a id='PCA'></a>

We used the first batch of 100.000 images to train our PCA model. Our model can be downloaded by executing this command in your terminal:    
```
wget -O PCA_model_300.pkl https://www.dropbox.com/s/8rqrxyiffx2hxvs/PCA_model_300.pkl?dl=0
```
The program will expect the model to be in the same folder as the full_vectors.npy and full_paths.npy so after downloading it place it there.


## NMSLIB installation
<a id='NMSLIB'></a>

Clone the github repository with this command:
```
git clone https://github.com/searchivarius/nmslib
```

Install prerequisite packages:
```
sudo apt-get install libboost-all-dev libgsl0-dev libeigen3-dev
```

Change directory to similarity_search and run the following commands:
```
cmake .
make  
```

Activate your virtual environment if you haven't done so already. Change directory to python bindings and run these commands:
```
python setup.py build
sudo python setup.py install
```

## Example results
<a id='Results'></a>

<table  border="0" width="100%" style="border:none">
<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/food.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/food_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/bird.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/bird_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/meadow.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/meadow_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/car.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/car_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/cat.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/cat_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/girl.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/girl_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/thumbnail_beach2.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/thumbnail_beach2_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/california.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/california_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/thumbnail_val_thorens.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/thumbnail_val_thorens_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/thumbnail_paris.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/thumbnail_paris_response.jpg" width="400">
</td>
</tr>

<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Query Image:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/thumbnail_street3.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Top 12 results:
<img src="https://github.com/Mungosin/AVSP/blob/master/results/thumbnail_street3_response.jpg" width="400">
</td>
</tr>
</table>
