# Testing First Machine Learning Code
from SimpleCV import *

hhfe = HueHistogramFeatureExtractor()
ehfe = EdgeHistogramFeatureExtractor()
haarfe = HaarLikeFeatureExtractor()
extractors = [hhfe,ehfe,haarfe]

tree = TreeClassifier(extractors)

trainPaths = ['./MLdata/Beer/train/','./MLdata/Wine/train/']
testPaths  = ['./MLdata/Beer/test/','./MLdata/Wine/test/']

classes = ['beer','wine']

#Train the data
print tree.train(trainPaths,classes,verbose=True)
print "----------------------------------------"
print tree.test(testPaths,classes,verbose=True)