#! /usr/bin/env python

import cvutils, moving, ml, storage, classifier
import yaml
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
parser.add_argument('-i', dest = 'videoFilename', help = 'name of the video file (overrides the configuration file)')
parser.add_argument('-n', dest = 'nObjects', help = 'number of objects to classify', type = int, default = None)
parser.add_argument('--start-frame0', dest = 'startFrame0', help = 'starts with first frame for videos with index problem where frames cannot be reached', action = 'store_true')
parser.add_argument('--plot-speed-distributions', dest = 'plotSpeedDistribution', help = 'simply plots the distributions used for each user type', action = 'store_true')
parser.add_argument('--max-speed-distribution-plot', dest = 'maxSpeedDistributionPlot', help = 'if plotting the user distributions, the maximum speed to display (km/h)', type = float, default = 50.)

args = parser.parse_args()
classifierInstance = classifier.createClassifier(args.classifierFilename)

if classifierInstance.classifierType == 'speed-classifier':
    objects = storage.loadTrajectoriesFromSqlite(args.databaseFilename, 'object', args.nObjects, withFeatures = True)
    for obj in objects:
        obj.setUserType(classifierInstance.classify(obj))
    #classifierInstance.save("saveSpeed.yaml")
    print('Saving user types')
    storage.setRoadUserTypes(args.databaseFilename, objects)
    sys.exit()

params, videoFilename, databaseFilename, invHomography, intrinsicCameraMatrix, distortionCoefficients, undistortedImageMultiplication, undistort, firstFrameNum = storage.processVideoArguments(args)
objects = storage.loadTrajectoriesFromSqlite(databaseFilename, 'object', args.nObjects, withFeatures = True)
#classifierParams = storage.ClassifierParameters(params.classifierFilename)
#classifierParams.convertToFrames(params.videoFrameRate, 3.6) # conversion from km/h to m/frame

if args.plotSpeedDistribution:
    if classifierInstance.classifierType == 'speedProbAppearance-classifier':
        classifierInstance.plotSpeedDistribut(args.maxSpeedDistributionPlot, params.videoFrameRate)
    else:
        print('not speedProbAppearanceClassifier, cannot plot. exiting')
    sys.exit()

timeInterval = moving.TimeInterval.unionIntervals([obj.getTimeInterval() for obj in objects])
if args.startFrame0:
    timeInterval.first = 0

capture = cv2.VideoCapture(videoFilename)
width = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

#if undistort: # setup undistortion
    #[map1, map2] = cvutils.computeUndistortMaps(width, height, undistortedImageMultiplication, intrinsicCameraMatrix, distortionCoefficients)
    #height, width = map1.shape

pastObjects = []
currentObjects = []
if capture.isOpened():
    ret = True
    frameNum = timeInterval.first
    if not args.startFrame0:
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frameNum)
    lastFrameNum = timeInterval.last

    while ret and frameNum <= lastFrameNum:
        ret, img = capture.read()
        if ret:
            if frameNum%50 == 0:
                print('frame number: {}'.format(frameNum))
            #if undistort:
                #img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            for obj in objects:
                if obj.getFirstInstant() <= frameNum: # if images are skipped
                    if classifierInstance.classifierType in ('thresSpeedAppearance-classifier', 'speedProbAppearance-classifier'):
                        classifierInstance.initClassify(obj)
                    currentObjects.append(obj)
                    objects.remove(obj)

            for obj in currentObjects:
                if obj.getLastInstant() <= frameNum:  # if images are skipped
                    classifierInstance.classify(obj)
                    pastObjects.append(obj)
                    currentObjects.remove(obj)
                else:
                    classifierInstance.classifyAtInstant(obj, img, frameNum, width, height)
        frameNum += 1
    
    for obj in currentObjects:
        classifierInstance.classify(obj)
        pastObjects.append(obj)

    '''if classifierInstance.classifierType == 'appearance-classifier':
        classifierInstance.save("saveAppearance.yaml")
    elif classifierInstance.classifierType == 'thresSpeedAppearance-classifier':
        classifierInstance.save("saveThresSpeedAppearance.yaml")
    elif classifierInstance.classifierType == 'speedProbAppearance-classifier':
        classifierInstance.save("saveSpeedProbAppearance.yaml")'''

    print('Saving user types')
    storage.setRoadUserTypes(databaseFilename, pastObjects)