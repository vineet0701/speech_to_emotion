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
from scipy.cluster.vq import kmeans, vq
from sklearn.mixture import GMM
from sklearn.externals.six.moves import xrange
import random
import winsound

emotion_array = ['happy', 'sadness', 'shame', 'neutral']

def getFeature(filename):
    (rate,sig) = wav.read(filename)
    mfcc_feat = mfcc(sig,rate, nfft=1250)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate, nfft=1250)
    #print fbank_feat
    #print mfcc_feat.shape, fbank_feat.shape
    features = np.hstack((mfcc_feat, fbank_feat, d_mfcc_feat))
    return features#np.array(performClustering(features))

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
        happyfeaturelist = []
        #count = 0
        for f in fileList:
            features = getFeature(directoryName + "/" + f)
            for feature in features:
                #count = count + 1
                happyfeaturelist.append(feature)
                
        happyfeaturelist = performClustering(happyfeaturelist, 50, 10)
        #featurelist, taglist = clusterFeatures(featurelist, taglist)
        directoryName = 'data/sadness'
        fileList = getFiles(directoryName)
        sadfeaturelist = []
        for f in fileList:
            features = getFeature(directoryName + "/" + f)
            for feature in features:
                sadfeaturelist.append(feature)
                #taglist.append(1)
        sadfeaturelist = performClustering(sadfeaturelist, 50, 10)

        for f in happyfeaturelist:
            featurelist.append(f)
            taglist.append(0)
        for f in sadfeaturelist:
            featurelist.append(f)
            taglist.append(1)

        # directoryName = '../../data_set/shame'
        # fileList = getFiles(directoryName)
        
        # for f in fileList:
        #     features = getFeature(directoryName + "/" + f)
        #     for feature in features:
        #         featurelist.append(feature)
        #         taglist.append(2)
        
        # directoryName = '../../data_set/neutral'
        # fileList = getFiles(directoryName)
        # for f in fileList:
        #     features = getFeature(directoryName + "/" + f)
        #     for feature in features:
        #         featurelist.append(feature)
        #         taglist.append(3)
        
        #featurelist, taglist = clusterFeatures(featurelist, taglist)
        featurelist = np.array(featurelist)
        taglist = np.array(taglist)
        print featurelist.shape, taglist.shape
        
        print taglist
        print(collections.Counter(taglist))
        print "Training model"
        clf = RandomForestClassifier(n_estimators=525)#, max_features = 'auto')
        clf = clf.fit(featurelist, taglist)
        #n_classes = 2
        #classifiers = dict((covar_type, GMM(n_components=n_classes, covariance_type=covar_type, init_params='wc', n_iter=20)
         #               for covar_type in ['spherical', 'diag', 'tied', 'full']))
        #clf = GMM(n_components=n_classes, covariance_type='full', init_params='wc', n_iter=20)
        #clf.means_ = np.array([featurelist[taglist == i].mean(axis=0)
                            #for i in xrange(n_classes)])
        #clf.fit(featurelist, taglist)
        
        print "Dumping model"
        joblib.dump(clf, 'data/model/model.pkl')
        print "Model dump successful"
    else:
        print "Loading model from memory"
        clf = joblib.load('data/model/model.pkl')
        print "Model load successful"
        test_audio = performClustering(getFeature('data/happy.wav'), 20, 2)
        testFeatureList = []
        for a in test_audio:
            testFeatureList.append(a)
        output = clf.predict(testFeatureList)
        counter=collections.Counter(output)
        print(counter)
        emotion_index = counter.most_common()[0][0]
        emotion_count = counter.most_common()[0][1]
        emotion = emotion_array[emotion_index]
        jobject = {}
        jobject['status'] = "OK"
        jobject['emotion'] = emotion
        jobject['emotion_index'] = str(emotion_index)
        jobject['confidence'] = str(float(emotion_count)/float(counter[0] + counter[1] + counter[2]))
        print(jobject)

def clusterFeatures(feature, tag):
    arr = np.array(feature).astype(float)
    centroid, _ = kmeans(arr, 30)
    marked, _ = vq(arr, centroid)
    cluster = {}
    for i in range(len(marked)):
        if marked[i] in cluster:
            cluster[marked[i]].append((arr[i],tag[i]))
        else:
            cluster[marked[i]] = [(arr[i], tag[i])]
    selectedVectors = []
    selectCount = 20
    for k in cluster.keys():
        l = cluster[k]
        if len(l) < selectCount:
            selectedVectors = selectedVectors + l
        else:
            selectedVectors = selectedVectors + [l[i] for i in random.sample(range(len(l)), selectCount)]
    features = []
    tags = []
    for vector in selectedVectors:
        features.append(vector[0])
        tags.append(vector[1])
    return features, tags

def performClustering(array, num_cluster = 20, sample_size=2):
    arr = np.array(array).astype(float)
    centroid, _ = kmeans(arr,num_cluster)
    # #print centroid
    # return np.array(centroid)
    marked,_ = vq(arr, centroid)
    clusterMap = {}
    for i in range(len(marked)):
        if marked[i] in clusterMap:
            clusterMap[marked[i]].append(arr[i])
        else:
            clusterMap[marked[i]] = [arr[i]]
    
    selectedVectors = []
    for k in clusterMap.keys():
        l = clusterMap[k]
        #print(len(l))
        if len(l) < sample_size:
            selectedVectors = selectedVectors + l
        else:
            selectedVectors = selectedVectors + [l[i] for i in random.sample(range(len(l)), sample_size)]
    return np.array(selectedVectors)


def GetAccuracy(output, tag):
    count = 0
    print "Comparing output and tag:"
    for i in xrange(len(output)):
        print output[i], tag[i]
        if output[i] == tag[i]:
            count = count + 1
    print float(count) / len(output)
    return float(count) / len(output)

def Beep():
    winsound.Beep(2500,1000)



#train(False)
#Beep()
train(True)
Beep()