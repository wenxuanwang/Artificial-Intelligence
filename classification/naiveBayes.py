# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# individual work by Wenxuan Wang 2165909

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    countLabel = util.Counter() 
    self.priors = util.Counter()
    probFeat = util.Counter()
    
    for label in self.legalLabels:
        countLabel[label] = util.Counter()
    
    for i in range(len(trainingData)): 
        self.priors[ trainingLabels[i]] += 1 
        temp = countLabel[ trainingLabels[i] ]
        for feature in self.features:
            temp[feature] += 1 if trainingData[i][feature] == 1 else 0
    
    for k in kgrid:
        temp = util.Counter()    
        for y in self.legalLabels:
            temp[y] = countLabel[y].copy()
                
        for y in self.legalLabels:
            temp[y].incrementAll(temp[y].keys(), k)
            temp[y].divideAll(self.priors[y]+2*k)
        probFeat[k] = temp
    
    if len(kgrid) < 0:
        self.probTable = probFeat[kgrid[0]] 
        return
    
    self.priors.normalize() 
    countMax = -1
    kFinal = -1
    for k in kgrid:
        self.probTable = probFeat[k]
        guess = self.classify(validationData)
        correct = [guess[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)  
        if countMax < correct:
            countMax = correct
            kFinal = k                                
    self.probTable = probFeat[kFinal]
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    for label in self.legalLabels:
        logJoint[label] = math.log(self.priors[label])
        for feature in self.features:
            logJoint[label] += math.log(self.probTable[label][feature]) if datum[feature] == 1 else math.log(1.0-self.probTable[label][feature])       
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    featOdd = util.Counter()
    for feature in self.features:
        featOdd[feature] = self.probTable[label1][feature] / self.probTable[label2][feature]
    featuresOdds = featOdd.sortedKeys()[0:100]
    return featuresOdds
    

    
      
