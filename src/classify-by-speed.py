#! /usr/bin/env python

import cvutils, moving, ml, storage, classifier

import numpy as np
import sys, argparse
#from cv2.ml import SVM_RBF, SVM_C_SVC
import cv2
from scipy.stats import norm, lognorm

# TODO add mode detection live, add choice of kernel and svm type (to be saved in future classifier format)

parser = argparse.ArgumentParser(description='The program processes indicators for all pairs of road users in the scene')
parser.add_argument('--cfg', dest = 'configFilename', help = 'name of the configuration file')
parser.add_argument('--cls', dest = 'classifierFilename', help = 'name of the classifier file', required = True)
parser.add_argument('-d', dest = 'databaseFilename', help = 'name of the Sqlite database file (overrides the configuration file)')
parser.add_argument('-n', dest = 'nObjects', help = 'number of objects to classify', type = int, default = None)

args = parser.parse_args()

#object is a list
objects = storage.loadTrajectoriesFromSqlite(args.databaseFilename, 'object', args.nObjects, withFeatures = True)
#features = storage.loadTrajectoriesFromSqlite(databaseFilename, 'feature')
speedclassifier = classifier.SpeedClassifier.load(args.classifierFilename)
for obj in objects:
    obj.setUserType(speedclassifier.classify(obj))

#speedclassifier.save("saveSpeed.yaml")
storage.setRoadUserTypes(args.databaseFilename, objects)
