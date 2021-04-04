import moving, storage, cvutils, ml, utils
import yaml
import sys
import numpy as np
from scipy.stats import norm, lognorm

def createClassifier(filename):
    with storage.openCheck(filename, 'r') as file:
        doc = yaml.load(file)
        if doc["classifierType"] == SpeedClassifier.classifierType:
            return SpeedClassifier.load(filename)
        if doc["descriptor"] == HOGSVMAppearanceClassifier.descriptor and doc["classifier"] == HOGSVMAppearanceClassifier.classifier:
            if doc["classifierType"] == HOGSVMAppearanceClassifier.classifierType:
                return HOGSVMAppearanceClassifier.load(filename)
            elif doc["classifierType"] == ThresSpeedAppearanceClassifier.classifierType:
                return ThresSpeedAppearanceClassifier.load(filename)
            elif doc["classifierType"] == SpeedProbAppearanceClassifier.classifierType:
                return SpeedProbAppearanceClassifier.load(filename)
    print('loading unknown classifierType. Exiting')
    sys.exit()

class Classifier(object):
    """docstring for Classifier"""
    def train(self):
        pass

    def load(self, filename):
        pass

    def save(self, filename):
        pass
    
    def classify(self, obj):
        pass

class SpeedClassifier(Classifier):
    """docstring for SpeedClassifier"""
    classifierType = 'speed-classifier'
    def __init__(self, userTypes = [], thresholds = [], aggregationFunc = 'median', nInstantsIgnoredAtEnds = 0, speedAggregationQuantile = 50):
        '''There should be n-1 thresholds for n userTypes, userTypes and thresholds are lists'''
        self.userTypes = userTypes 
        self.thresholds = thresholds 
        self.nInstantsIgnoredAtEnds = nInstantsIgnoredAtEnds
        self.speedAggregationQuantile = speedAggregationQuantile
        self.classifierType = SpeedClassifier.classifierType

        if aggregationFunc == 'median':
            self.aggregationFunc = np.median
        elif aggregationFunc == 'mean':
            self.aggregationFunc = np.mean
        elif aggregationFunc == 'quantile':
            self.aggregationFunc = lambda speeds: np.percentile(speeds, self.speedAggregationQuantile)
            self.aggregationFunc2 = np.percentile
        else:
            print('loading unknown speed aggregation method: {}. Exiting'.format(aggregationFunc))
            sys.exit()

    def classify(self, obj):
        '''If smaller than threshold, return the smaller type. 
        Larger than the max threshold then return the max type'''
        x = 0
        aggrSpeed = self.aggregationFunc(obj.getSpeeds(self.nInstantsIgnoredAtEnds))
        for elem in self.thresholds:
            if aggrSpeed < elem:
                return self.userTypes[x]
            x += 1
        return self.userTypes[x]
    
    @staticmethod    
    def load(filename):
        with open(filename, 'r') as speedFile:
            doc = yaml.load(speedFile)
            return SpeedClassifier(doc["userTypes"], doc["thresholds"], doc["aggregationFunc"], doc["nInstantsIgnoredAtEnds"], doc["speedAggregationQuantile"])
        
    def train(self):
        pass

    def save(self, filename):
        if self.aggregationFunc == np.median:
            strAggregationFunc = 'median'
        elif self.aggregationFunc == np.mean:
            strAggregationFunc = 'mean' 
        #can't save self.aggregationFunc == lambda...
        elif self.aggregationFunc2 == np.percentile:
            strAggregationFunc = 'quantile'
        else:
            print('saving unknown speed aggregation method: {}. Exiting'.format(self.aggregationFunc))
            sys.exit()

        data = dict(
            userTypes = self.userTypes,
            thresholds = self.thresholds,
            aggregationFunc = strAggregationFunc,
            nInstantsIgnoredAtEnds = self.nInstantsIgnoredAtEnds,
            speedAggregationQuantile = self.speedAggregationQuantile,
            classifierType = SpeedClassifier.classifierType
            )

        with storage.openCheck(filename, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style = False)

class HOGSVMAppearanceClassifier(Classifier):
    """docstring for HOGSVMAppearanceClassifier"""
    classifierType = 'appearance-classifier'
    descriptor = 'hog'
    classifier = 'svm'
    def __init__(self, userTypes, SVMFilename, percentIncreaseCrop, minNPixels, rescaleSize, orientations, pixelsPerCell, cellsPerBlock, blockNorm):
        self.userTypes = userTypes
        self.SVMFilename = SVMFilename
        self.percentIncreaseCrop = percentIncreaseCrop
        self.minNPixels = minNPixels
        self.rescaleSize = rescaleSize
        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.blockNorm = blockNorm
        self.classifierType = HOGSVMAppearanceClassifier.classifierType
        self.descriptor = HOGSVMAppearanceClassifier.descriptor
        self.classifier = HOGSVMAppearanceClassifier.classifier

        if self.SVMFilename is not None:
            self.appearanceClassifier = ml.SVM()
            self.appearanceClassifier.load(self.SVMFilename)

    def classifyAtInstant(self, obj, img, instant, width, height):
        croppedImg = cvutils.imageBox(img, obj, instant, width, height, self.percentIncreaseCrop, self.percentIncreaseCrop, self.minNPixels)
        if not hasattr(obj, 'userTypes'):
            #setattr(obj, userTypes, {})
            obj.userTypes = {}
        if croppedImg is not None and len(croppedImg) > 0:
            hog = cvutils.HOG(croppedImg, (self.rescaleSize, self.rescaleSize), self.orientations, (self.pixelsPerCell, self.pixelsPerCell), (self.cellsPerBlock, self.cellsPerBlock), self.blockNorm, visualize=False, normalize=False)
            obj.userTypes[instant] = int(self.appearanceClassifier.predict(hog))
        else:
            obj.userTypes[instant] = 0
    
    def classify(self, obj, width = 0, height = 0, images = None):
        if len(obj.userTypes) != obj.length() and images is not None: # if classification has not been done previously
            for t in obj.getTimeInterval():
                if t not in obj.userTypes:
                    self.classifyAtInstant(obj, images[t], t, width, height)
        # result is P(Class|Appearance)
        nInstantsUserType = {userTypeNum: 0 for userTypeNum in self.userTypes}# number of instants the object is classified as userTypename
        for t in obj.userTypes:
            nInstantsUserType[obj.userTypes[t]] = nInstantsUserType.get(obj.userTypes[t], 0) + 1
        # class is the user type that maximizes nInstantsUserType
        return utils.argmaxDict(nInstantsUserType)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as appearanceFile:
            doc = yaml.load(appearanceFile)
            return HOGSVMAppearanceClassifier(doc["userTypes"], doc["pbvSVMFilename"], doc["percentIncreaseCrop"], doc["minNPixelsCrop"], doc["rescaleSize"], doc["nOrientations"], doc["nPixelsCell"], doc["nCellsBlock"], doc["blockNorm"])

    def train(self, directoryName, svmType, kernelType, degree, gamma, coef0, cvalue, nu, svmP): 
        imageDirectories = {'pedestrian': directoryName + "/Pedestrians/",
                            'bicycle': directoryName + "/Cyclists/",
                            'car': directoryName + "/Vehicles/"}

        trainingSamplesPBV = {}
        trainingLabelsPBV = {}
        trainingSamplesBV = {}
        trainingLabelsBV = {}
        trainingSamplesPB = {}
        trainingLabelsPB = {}
        trainingSamplesPV = {}
        trainingLabelsPV = {}

        for k, v in imageDirectories.iteritems():
            print('Loading {} samples'.format(k))
            trainingSamples, trainingLabels = cvutils.createHOGTrainingSet(v, moving.userType2Num[k], (self.rescaleSize, self.rescaleSize), self.orientations, (self.pixelsPerCell, self.pixelsPerCell), self.blockNorm, (self.cellsPerBlock, self.cellsPerBlock))
            trainingSamplesPBV[k], trainingLabelsPBV[k] = trainingSamples, trainingLabels
            if k != 'pedestrian':
                trainingSamplesBV[k], trainingLabelsBV[k] = trainingSamples, trainingLabels
            if k != 'car':
                trainingSamplesPB[k], trainingLabelsPB[k] = trainingSamples, trainingLabels
            if k != 'bicycle':
                trainingSamplesPV[k], trainingLabelsPV[k] = trainingSamples, trainingLabels

        # Training the Support Vector Machine
        print "Training Pedestrian-Cyclist-Vehicle Model"
        model = ml.SVM(svmType, kernelType, degree, gamma, coef0, cvalue, nu, svmP)
        model.train(np.concatenate(trainingSamplesPBV.values()), np.concatenate(trainingLabelsPBV.values()))
        model.save(directoryName + "/modelPBV.xml")

        print "Training Cyclist-Vehicle Model"
        model = ml.SVM(svmType, kernelType, degree, gamma, coef0, cvalue, nu, svmP)
        model.train(np.concatenate(trainingSamplesBV.values()), np.concatenate(trainingLabelsBV.values()))
        model.save(directoryName + "/modelBV.xml")

        print "Training Pedestrian-Cyclist Model"
        model = ml.SVM(svmType, kernelType, degree, gamma, coef0, cvalue, nu, svmP)
        model.train(np.concatenate(trainingSamplesPB.values()), np.concatenate(trainingLabelsPB.values()))
        model.save(directoryName + "/modelPB.xml")

        print "Training Pedestrian-Vehicle Model"
        model = ml.SVM(svmType, kernelType, degree, gamma, coef0, cvalue, nu, svmP)
        model.train(np.concatenate(trainingSamplesPV.values()), np.concatenate(trainingLabelsPV.values()))
        model.save(directoryName + "/modelPV.xml")

    def save(self, filename):
        data = dict(
            userTypes = self.userTypes,
            pbvSVMFilename = self.SVMFilename,
            percentIncreaseCrop = self.percentIncreaseCrop,
            minNPixelsCrop = self.minNPixels,
            rescaleSize = self.rescaleSize,
            nOrientations = self.orientations,
            nPixelsCell = self.pixelsPerCell,
            nCellsBlock = self.cellsPerBlock,
            blockNorm = self.blockNorm,
            classifierType = HOGSVMAppearanceClassifier.classifierType,
            descriptor = HOGSVMAppearanceClassifier.descriptor,
            classifier = HOGSVMAppearanceClassifier.classifier
            )

        with storage.openCheck(filename, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style = False)

class ThresSpeedAppearanceClassifier(HOGSVMAppearanceClassifier):
    classifierType = 'thresSpeedAppearance-classifier'
    def __init__(self, userTypes, bvSVMFilename, pbvSVMFilename, percentIncreaseCrop, minNPixels, rescaleSize, orientations, pixelsPerCell, cellsPerBlock, blockNorm, aggregationFunc = 'median', pedBikeSpeedThreshold = float('Inf'), bikeCarSpeedThreshold = float('Inf'), nInstantsIgnoredAtEnds = 0, speedAggregationQuantile = 50):
        super(ThresSpeedAppearanceClassifier, self).__init__(userTypes, None, percentIncreaseCrop, minNPixels, rescaleSize, orientations, pixelsPerCell, cellsPerBlock, blockNorm) 
        self.bvSVMFilename = bvSVMFilename
        self.pbvSVMFilename = pbvSVMFilename
        self.pedBikeSpeedThreshold = pedBikeSpeedThreshold
        self.bikeCarSpeedThreshold = bikeCarSpeedThreshold
        self.nInstantsIgnoredAtEnds = nInstantsIgnoredAtEnds
        self.speedAggregationQuantile = speedAggregationQuantile
        self.appearanceClassifier = ml.SVM()
        self.classifierType = ThresSpeedAppearanceClassifier.classifierType

        if aggregationFunc == 'median':
            self.aggregationFunc = np.median
        elif aggregationFunc == 'mean':
            self.aggregationFunc = np.mean
        elif aggregationFunc == 'quantile':
            self.aggregationFunc = lambda speeds: np.percentile(speeds, self.speedAggregationQuantile)
            self.aggregationFunc2 = np.percentile
        else:
            print('loading unknown speed aggregation method: {}. Exiting'.format(aggregationFunc))
            sys.exit()

    def initClassify(self, obj):
        obj.aggregatedSpeed = self.aggregationFunc(obj.getSpeeds(self.nInstantsIgnoredAtEnds))
        if obj.aggregatedSpeed < self.pedBikeSpeedThreshold or self.bvSVMFilename is None:
            self.appearanceClassifier.load(self.pbvSVMFilename)
        elif obj.aggregatedSpeed < self.bikeCarSpeedThreshold:
            self.appearanceClassifier.load(self.bvSVMFilename)
        else:
            class CarClassifier:
                def predict(self, hog):
                    return moving.userType2Num['car']
            self.appearanceClassifier = CarClassifier()

    @staticmethod    
    def load(filename):
        with open(filename, 'r') as thresSpeedAppearanceFile:
            doc = yaml.load(thresSpeedAppearanceFile)
            return ThresSpeedAppearanceClassifier(doc["userTypes"], doc["bvSVMFilename"], doc["pbvSVMFilename"], doc["percentIncreaseCrop"], doc["minNPixelsCrop"], doc["rescaleSize"], doc["nOrientations"], doc["nPixelsCell"], doc["nCellsBlock"], doc["blockNorm"], doc["aggregationFunc"], doc["pedBikeSpeedThreshold"], doc["bikeCarSpeedThreshold"], doc["nInstantsIgnoredAtEnds"])
 
    def train(self): 
        pass

    def save(self, filename):
        if self.aggregationFunc == np.median:
            strAggregationFunc = 'median'
        elif self.aggregationFunc == np.mean:
            strAggregationFunc = 'mean' 
        #can't save self.aggregationFunc == lambda...
        elif self.aggregationFunc2 == np.percentile:
            strAggregationFunc = 'quantile'
        else:
            print('saving unknown speed aggregation method: {}. Exiting'.format(self.aggregationFunc))
            sys.exit()

        data = dict(
            userTypes = self.userTypes,
            aggregationFunc = strAggregationFunc,
            pbvSVMFilename = self.pbvSVMFilename,
            bvSVMFilename = self.bvSVMFilename, 
            percentIncreaseCrop = self.percentIncreaseCrop,
            minNPixelsCrop = self.minNPixels,
            rescaleSize = self.rescaleSize,
            nOrientations = self.orientations,
            nPixelsCell = self.pixelsPerCell,
            speedAggregationQuantile = self.speedAggregationQuantile,
            nCellsBlock = self.cellsPerBlock,
            pedBikeSpeedThreshold = self.pedBikeSpeedThreshold,
            bikeCarSpeedThreshold = self.bikeCarSpeedThreshold,
            nInstantsIgnoredAtEnds = self.nInstantsIgnoredAtEnds,
            blockNorm = self.blockNorm,
            classifierType = ThresSpeedAppearanceClassifier.classifierType,
            descriptor = HOGSVMAppearanceClassifier.descriptor,
            classifier = HOGSVMAppearanceClassifier.classifier
            )

        with storage.openCheck(filename, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style = False)

class SpeedProbAppearanceClassifier(ThresSpeedAppearanceClassifier):
    classifierType = 'speedProbAppearance-classifier'
    def __init__(self, userTypes, meanVehicleSpeed, stdVehicleSpeed, meanPedestrianSpeed, stdPedestrianSpeed, scaleCyclistSpeed, locationCyclistSpeed, minSpeedEquiprobable, maxPercentUnknown, bvSVMFilename, pbvSVMFilename, percentIncreaseCrop, minNPixels, rescaleSize, orientations, pixelsPerCell, cellsPerBlock, blockNorm, aggregationFunc = 'median', pedBikeSpeedThreshold = float('Inf'), bikeCarSpeedThreshold = float('Inf'), nInstantsIgnoredAtEnds = 0, speedAggregationQuantile = 50):
        super(SpeedProbAppearanceClassifier, self).__init__(userTypes, bvSVMFilename, pbvSVMFilename, percentIncreaseCrop, minNPixels, rescaleSize, orientations, pixelsPerCell, cellsPerBlock, blockNorm, aggregationFunc, pedBikeSpeedThreshold, bikeCarSpeedThreshold, nInstantsIgnoredAtEnds, speedAggregationQuantile)
        self.minSpeedEquiprobable = minSpeedEquiprobable
        self.maxPercentUnknown = maxPercentUnknown
        self.meanVehicleSpeed = meanVehicleSpeed
        self.stdVehicleSpeed = stdVehicleSpeed
        self.meanPedestrianSpeed = meanPedestrianSpeed
        self.stdPedestrianSpeed = stdPedestrianSpeed
        self.scaleCyclistSpeed = scaleCyclistSpeed
        self.locationCyclistSpeed = locationCyclistSpeed
        self.userTypes2 = userTypes
        self.classifierType = SpeedProbAppearanceClassifier.classifierType

        carNorm = norm(self.meanVehicleSpeed, self.stdVehicleSpeed)
        pedNorm = norm(self.meanPedestrianSpeed, self.stdPedestrianSpeed)
        #numpy lognorm shape, loc, scale: shape for numpy is scale (std of the normal) and scale for numpy is exp(location) (loc=mean of the normal)
        bicLogNorm = lognorm(self.scaleCyclistSpeed, loc = 0., scale = np.exp(self.locationCyclistSpeed))
        self.speedProbabilities = {'car': lambda s: carNorm.pdf(s),
                                   'pedestrian': lambda s: pedNorm.pdf(s), 
                                   'bicycle': lambda s: bicLogNorm.pdf(s)}

    def classify(self, obj, width = 0, height = 0, images = None):
        if not hasattr(obj, 'aggregatedSpeed') or not hasattr(obj, 'userTypes'):
            print('Initializing the data structures for classification by HoG-SVM')
            self.initClassify(obj)
        if len(obj.userTypes) != obj.length() and images is not None: # if classification has not been done previously
            for t in obj.getTimeInterval():
                if t not in obj.userTypes:
                    self.classifyAtInstant(obj, images[t], t, width, height)
        # compute P(Speed|Class)
        if self.speedProbabilities is not None and obj.aggregatedSpeed < self.minSpeedEquiprobable: # equiprobable information from speed
            self.userTypes = {moving.userType2Num[userTypename]: self.speedProbabilities[userTypename](obj.aggregatedSpeed) for userTypename in self.speedProbabilities}
        # compute P(Class|Appearance)
        nInstantsUserType = {userTypeNum: 0 for userTypeNum in self.userTypes}# number of instants the object is classified as userTypename
        nInstantsUserType[moving.userType2Num['unknown']] = 0
        for t in obj.userTypes:
            nInstantsUserType[obj.userTypes[t]] = nInstantsUserType.get(obj.userTypes[t], 0) + 1
        # result is P(Class|Appearance) x P(Speed|Class)
        if nInstantsUserType[moving.userType2Num['unknown']] < self.maxPercentUnknown*obj.length(): # if not too many unknowns
            for userTypeNum in self.userTypes:
                self.userTypes[userTypeNum] *= nInstantsUserType[userTypeNum]
        # class is the user type that maximizes self.userTypes
        if nInstantsUserType[moving.userType2Num['unknown']] >= self.maxPercentUnknown*obj.length() and (self.speedProbabilities is None or obj.aggregatedSpeed < self.minSpeedEquiprobable): # if no speed information and too many unknowns
            return moving.userType2Num['unknown']
        else:
            return utils.argmaxDict(self.userTypes)

    def plotSpeedDistribut(self, maxSpeedDistributionPlot, videoFrameRate):
        import matplotlib.pyplot as plt
        plt.figure()
        for k in self.speedProbabilities:
            plt.plot(np.arange(0.1, maxSpeedDistributionPlot, 0.1), [self.speedProbabilities[k](s/(3.6*videoFrameRate)) for s in np.arange(0.1, maxSpeedDistributionPlot, 0.1)], label = k)
        maxProb = -1.
        for k in self.speedProbabilities:
            maxProb = max(maxProb, np.max([self.speedProbabilities[k](s/(3.6*videoFrameRate)) for s in np.arange(0.1, maxSpeedDistributionPlot, 0.1)]))
        plt.plot([self.minSpeedEquiprobable*3.6*videoFrameRate]*2, [0., maxProb], 'k-')
        plt.text(self.minSpeedEquiprobable*3.6*videoFrameRate, maxProb, 'threshold for equiprobable class')
        plt.xlabel('Speed (km/h)')
        plt.ylabel('Probability')
        plt.legend()
        plt.title('Probability Density Function')
        plt.show()

    @staticmethod    
    def load(filename):
        with open(filename, 'r') as speedProbAppearanceFile:
            doc = yaml.load(speedProbAppearanceFile)
            return SpeedProbAppearanceClassifier(doc["userTypes"], doc["meanVehicleSpeed"], doc["stdVehicleSpeed"], doc["meanPedestrianSpeed"], doc["stdPedestrianSpeed"], doc["scaleCyclistSpeed"], doc["locationCyclistSpeed"], doc["minSpeedEquiprobable"], doc["maxPercentUnknown"], doc["bvSVMFilename"], doc["pbvSVMFilename"], doc["percentIncreaseCrop"], doc["minNPixelsCrop"], doc["rescaleSize"], doc["nOrientations"], doc["nPixelsCell"], doc["nCellsBlock"], doc["blockNorm"], doc["aggregationFunc"], doc["pedBikeSpeedThreshold"], doc["bikeCarSpeedThreshold"], doc["nInstantsIgnoredAtEnds"])
 
    def train(self): 
        pass

    def save(self, filename):
        if self.aggregationFunc == np.median:
            strAggregationFunc = 'median'
        elif self.aggregationFunc == np.mean:
            strAggregationFunc = 'mean' 
        #can't save self.aggregationFunc == lambda...
        elif self.aggregationFunc2 == np.percentile:
            strAggregationFunc = 'quantile'
        else:
            print('saving unknown speed aggregation method: {}. Exiting'.format(self.aggregationFunc))
            sys.exit()

        data = dict(
            userTypes = self.userTypes2,
            meanVehicleSpeed = self.meanVehicleSpeed,
            stdVehicleSpeed = self.stdVehicleSpeed,
            meanPedestrianSpeed = self.meanPedestrianSpeed,
            stdPedestrianSpeed = self.stdPedestrianSpeed,
            scaleCyclistSpeed = self.scaleCyclistSpeed,
            locationCyclistSpeed = self.locationCyclistSpeed,
        	minSpeedEquiprobable = self.minSpeedEquiprobable,
        	maxPercentUnknown = self.maxPercentUnknown,
            aggregationFunc = strAggregationFunc,
            speedAggregationQuantile = self.speedAggregationQuantile,
            pbvSVMFilename = self.pbvSVMFilename,
            bvSVMFilename = self.bvSVMFilename, 
            percentIncreaseCrop = self.percentIncreaseCrop,
            minNPixelsCrop = self.minNPixels,
            rescaleSize = self.rescaleSize,
            nOrientations = self.orientations,
            nPixelsCell = self.pixelsPerCell,
            nCellsBlock = self.cellsPerBlock,
            pedBikeSpeedThreshold = self.pedBikeSpeedThreshold,
            bikeCarSpeedThreshold = self.bikeCarSpeedThreshold,
            nInstantsIgnoredAtEnds = self.nInstantsIgnoredAtEnds,
            blockNorm = self.blockNorm,
            classifierType = SpeedProbAppearanceClassifier.classifierType,
            descriptor = HOGSVMAppearanceClassifier.descriptor,
            classifier = HOGSVMAppearanceClassifier.classifier
            )

        with storage.openCheck(filename, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style = False)

if __name__ == "__main__":
    import doctest
    import unittest
    suite = doctest.DocFileSuite('classifier.txt')
    unittest.TextTestRunner().run(suite)