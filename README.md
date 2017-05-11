# AVSP
Analysis of Big Datasets

## How to get everything setup

### Getting the dataset
The code will work on any dataset, it expects to get a root folder which will contain only images and/or folders with images

The dataset we used was the OpenImage dataset and we used a downloader from this github https://github.com/ejlb/google-open-image-download 
to download it effectively. 

After u clone the repository and get the CSV files you execute this in your terminal:  
python2 download.py [CSV_PATH] [OUTPUT_FOLDER_PATH] 

### Tensorflow models
Open the terminal and change the directory to the project root folder and run this in the terminal:  
git clone https://github.com/tensorflow/models/

When the repository downloads, get the required checkpoint by running  
wget -O InceptionV4.tar.gz http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz

Untar the downloaded file by executing:  
tar -xvf InceptionV4.tar.gz

### Python packages
If you don't have an NVIDIA GPU then change the tensorflow-gpu in requirements.txt to just tensorflow before running the installation

Now position yourself in the project root folder

To use the code we suggest creating a virtual environment
First install virtual environment with:  
pip install virtualenv

After that create your virtual environment:  
virtualenv -p /usr/bin/python2 env

Activate the environment with:  
source ./env/bin/activate

Install the prerequisite packages  
pip2 install -r requirements.txt

After this install either ipython2 or jupyter notebook and install the required kernel so you can access the environment
You should be ready to go with just executing this line in the terminal:  
python -m ipykernel install --user --name=[Name you desire]

To get the progress bar to work in the notebook run this command in your terminal  
jupyter nbextension enable --py --sys-prefix widgetsnbextension

Now you're ready to run the notebook, in the terminal enter:  
ipython2 notebook
