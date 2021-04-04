#! /usr/bin/env python

import argparse
from cv2 import SVM_RBF, SVM_C_SVC
#from cv2.ml import SVM_RBF, SVM_C_SVC, ROW_SAMPLE # row_sample for layout in cv2.ml.SVM_load
import classifier

parser = argparse.ArgumentParser(description='The program processes indicators for all pairs of road users in the scene')
parser.add_argument('-d', dest = 'directoryName', help = 'parent directory name for the directories containing the samples for the different road users', required = True)
parser.add_argument('--kernel', dest = 'kernelType', help = 'kernel type for the support vector machine (SVM)', default = SVM_RBF, type = long)
parser.add_argument('--svm', dest = 'svmType', help = 'SVM type', default = SVM_C_SVC, type = long)
parser.add_argument('--deg', dest = 'degree', help = 'SVM degree', default = 0, type = int)
parser.add_argument('--gamma', dest = 'gamma', help = 'SVM gamma', default = 1, type = int)
parser.add_argument('--coef0', dest = 'coef0', help = 'SVM coef0', default = 0, type = int)
parser.add_argument('--cvalue', dest = 'cvalue', help = 'SVM Cvalue', default = 1, type = int)
parser.add_argument('--nu', dest = 'nu', help = 'SVM nu', default = 0, type = int)
parser.add_argument('--svmp', dest = 'svmP', help = 'SVM p', default = 0, type = int)
parser.add_argument('--cls', dest = 'classifierFilename', help = 'name of the classifier file', required = True)

args = parser.parse_args()
classifierInstance = classifier.createClassifier(args.classifierFilename)
classifierInstance.train(args.directoryName, args.svmType, args.kernelType, args.degree, args.gamma, args.coef0, args.cvalue, args.nu, args.svmP)

