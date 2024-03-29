# -*- coding: utf-8 -*-
"""RCNN Trainer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bI0tHvV1RSm_cWXbUsles7LmL5M4jKym

#Ensure Version is Correct
"""

from google.colab import drive
drive.mount('/content/drive')


import tensorflow as tf
if tf.__version__ == '1.14.0':
  print('Version Check passed')
else:
  print('Incompatible version')
  input('Interrupt')

"""#Compling models and Testing"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content
!git clone --quiet https://github.com/tensorflow/models.git
!apt-get install -qq protobuf-compiler python-pil python-lxml python-tk
!pip install -q Cython contextlib2 pillow lxml matplotlib
!pip install -q pycocotools
!pip install pandas
!pip install opencv-python

# %cd /content/models/research
!protoc object_detection/protos/*.proto --python_out=.

import os
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/'
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/'

!python object_detection/builders/model_builder_test.py

"""#Download Model"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/models/research/object_detection
!wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
!tar -xvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz

"""#Copy Dataset and API Tutorial"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content
!rm -r ObjDet_Demo
!git clone --quiet https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10.git
!mv /content/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 /content/ObjDet_Demo

# %cd /content/ObjDet_Demo
!rm -r training/*
!rm -r inference_graph/*
!rm -r images/*.*
!rm -r images/test
!rm -r images/train

# %cd /content/drive/"My Drive"/Images
!cp -rv test /content/ObjDet_Demo/images/test
!cp -rv train /content/ObjDet_Demo/images/train

!mv -v /content/ObjDet_Demo/* /content/models/research/object_detection/

"""#Build and Install setup.py"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/models/research
!python setup.py build
!python setup.py install

"""#Generate CSV, TfRecords and Label Map"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/models/research/object_detection
!python xml_to_csv.py

# %cd /content/models/research/object_detection
!rm -r generate_tfrecord.py
# %cd /content/drive/My Drive/Modified Files
!cp generate_tfrecord.py /content/models/research/object_detection/
# %cd /content/models/research/object_detection
!python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
!python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

# %cd /content/models/research/object_detection/training
!rm -r faster_rcnn_resnet101_coco.config
# %cd /content/drive/My Drive/Modified Files
!cp faster_rcnn_resnet101_coco.config /content/models/research/object_detection/training/
!cp label_map.pbtxt /content/models/research/object_detection/training
# %cd /content/models/research/object_detection/training
!mv -v label_map.pbtxt labelmap.pbtxt

"""#Copy legacy file and begin training"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/models/research/object_detection/legacy
!cp train.py /content/models/research/object_detection/
# %cd /content/models/research/object_detection/
!python train.py --logtostderr --train_dir=/content/models/research/object_detection/training/ --pipeline_config_path=/content/models/research/object_detection/training/faster_rcnn_resnet101_coco.config

"""#Create frozen inference graph"""

chkNum = input("Change Checkpoint Number:")
!python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix training/model.ckpt-2823 --output_directory inference_graph

"""#Copy frozen inference graph to Google Drive"""

!cp -r inference_graph /content/drive/"My Drive"/Saved_Models/Faster_RCNN/