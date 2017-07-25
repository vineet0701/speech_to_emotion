import os
from os import listdir
from os.path import isfile, join
import json
import numpy as np
import collections
import traceback
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib


def getFeature(filename):
    (rate,sig) = wav.read(filename)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    features = np.hstack((mfcc_feat, d_mfcc_feat, fbank_feat))
    return features

def getFiles(directoryName):
    onlyFiles = [f for f in listdir(directoryName) if isfile(join(directoryName, f))]
    return onlyFiles


def train(flag=False):
    
    ## Method 1 ##
    # clf = RandomForestClassifier(n_estimators = 150, max_features = 'auto')
    # print np.mean(cross_val_score(clf, featurelist, taglist, cv = 10))


    ## Method 2 ##
    # X_train, X_test, Y_train, Y_test = train_test_split(featurelist, taglist, test_size = 0.2, random_state = 0)
    # clf = RandomForestClassifier(n_estimators=150, max_features = 'auto')
    # clf = clf.fit(X_train, Y_train)
    # output = clf.predict(X_test)
    # accuracy = GetAccuracy(output, Y_test)
    
    clf = None
    ## Method 3 ##
    if not flag:
        directoryName = 'data/happy'
        fileList = getFiles(directoryName)
        featurelist = []
        taglist = []
        for f in fileList:
            features = getFeature(directoryName + "/" + f)
            for feature in features:
                featurelist.append(feature)
                taglist.append(0)

        directoryName = 'data/sadness'
        fileList = getFiles(directoryName)
        for f in fileList:
            features = getFeature(directoryName + "/" + f)
            for feature in features:
                featurelist.append(feature)
                taglist.append(1)
        
        # directoryName = 'data/panic'
        # fileList = getFiles(directoryName)
        
        # for f in fileList:
        #     features = getFeature(directoryName + "/" + f)
        #     for feature in features:
        #         featurelist.append(feature)
        #         taglist.append(2)
        
        directoryName = 'data/angry'
        fileList = getFiles(directoryName)
        for f in fileList:
            features = getFeature(directoryName + "/" + f)
            for feature in features:
                featurelist.append(feature)
                taglist.append(2)

        featurelist = np.array(featurelist)
        taglist = np.array(taglist)
        print featurelist.shape, taglist.shape
        
        print taglist

        print "Training model"
        clf = RandomForestClassifier(n_estimators=150, max_features = 'auto')
        clf = clf.fit(featurelist, taglist)
        print "Dumping model"
        joblib.dump(clf, 'data/model/model.pkl')
        print "Model dump successful"
    else:
        print "Loading model from memory"
        clf = joblib.load('data/model/model.pkl')
        print "Model load successful"
        test_audio = getFeature('data/angry.wav')
        testFeatureList = []
        for a in test_audio:
            testFeatureList.append(a)
        output = clf.predict(testFeatureList)
        counter=collections.Counter(output)
        print sorted(counter.elements())[0]

def GetAccuracy(output, tag):
    count = 0
    print "Comparing output and tag:"
    for i in xrange(len(output)):
        print output[i], tag[i]
        if output[i] == tag[i]:
            count = count + 1
    print float(count) / len(output)
    return float(count) / len(output)


train(True)