#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:35:45 2017

@author: asankar
"""
from alexnet import AlexNet
import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np
import h5py
from keras import backend as K


dbpath = '../TORCS_Training_1F/'
db = plyvel.DB(dbpath)
print(len(db))
    #keys = []
    #for key, value in db:
    #    keys.append(key)