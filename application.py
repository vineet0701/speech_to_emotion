#!/usr/bin/env python

import os
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from scipy.cluster.vq import kmeans, vq
import json
from sklearn.externals import joblib
import numpy as np
import collections
import traceback
import subprocess
from pydub import AudioSegment
import random

from flask import Flask, request

application = Flask(__name__)

emotion_array = ['happy', 'sadness', 'angry']

def convertFile(filename):
    file_format = filename.split('.')[-1]
    if file_format == "aac":
        sound = AudioSegment.from_file(filename, "aac")
        sound.export("./inputFile.wav", format="wav")
    elif file_format == "3gpp":
        sound = AudioSegment.from_file(filename, "3gp")
        sound.export("/inputFile.wav", format="wav")
    elif file_format == "mp3":
        sound = AudioSegment.from_mp3(filename)
        sound.export("/inputFile.wav", format="wav")
    else:
        return filename
    return "inputFile.wav"

def allowed_file(filename):
    return True

def secure_filename(filename):
    col = filename.split('/')
    filename = col[-1]
    return filename

@application.route('/testPath')
def test_path():
    return '<h1 align="center">This is a test link</h>'

@application.route('/')
def landing_page():
    return '<h1 align="center">Welcome to speech to emotion prediction API</h1>'

@application.route('/getEmotion', methods=['GET','POST'])
def get_emotion():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                print(filename)
                file.save(os.path.join("./", filename))
                print("file saved")
                filename = convertFile("./" + filename)
                print("Converted File: " + filename)
                
                clf = joblib.load('data/model/model.pkl')
                predictions = []
                for i in range(10):
                    test_audio = performClustering(getFeature(os.path.join("./", filename)), 20, 2)
                    #print("Preparing file")
                    testFeatureList = []
                    for a in test_audio:
                        testFeatureList.append(a)
                    output = clf.predict(testFeatureList)
                    counter=collections.Counter(output)
                    print("Prediction: ", counter)
                    emotion_index = counter.most_common()[0][0]
                    predictions.append(emotion_index)
                counter = collections.Counter(predictions)
                print(counter)
                emotion_index = counter.most_common()[0][0]
                emotion_count = counter.most_common()[0][1]
                emotion = emotion_array[emotion_index]
                jobject = {}
                jobject['status'] = "OK"
                jobject['emotion'] = emotion
                jobject['emotion_index'] = str(emotion_index)
                jobject['confidence'] = str(float(emotion_count)/float(counter[0] + counter[1] + counter[2]))
                return json.dumps(jobject)
            except Exception as ex:
                jobject = {}
                jobject['status'] = "Error"
                jobject['Message'] = traceback.format_exc()
                print(jobject)
                return json.dumps(jobject)
        else:
            jobject = {}
            jobject['status'] = "Erorr"
            jobject['Message'] = "Check file or the format"
            return json.dumps(jobject)
    else:
        jobject = {}
        jobject['status'] = "Error"
        jobject['Message'] = "Not a proper POST request"
        return json.dumps(jobject)

def getFeature(filename):
    (rate,sig) = wav.read(filename)
    mfcc_feat = mfcc(sig,rate, nfft=1250)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate, nfft=1250)
    #print fbank_feat
    #print mfcc_feat.shape, fbank_feat.shape
    features = np.hstack((mfcc_feat, fbank_feat, d_mfcc_feat))
    return features#np.array(performClustering(features))

def performClustering(array, num_cluster = 20, sample_size=2):
    arr = np.array(array).astype(float)
    centroid, _ = kmeans(arr,num_cluster)
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
        if len(l) < sample_size:
            selectedVectors = selectedVectors + l
        else:
            selectedVectors = selectedVectors + [l[i] for i in random.sample(range(len(l)), sample_size)]
    return np.array(selectedVectors)

# def getFeature(filename):
#     (rate,sig) = wav.read(filename)
#     mfcc_feat = mfcc(sig,rate)
#     d_mfcc_feat = delta(mfcc_feat, 2)
#     fbank_feat = logfbank(sig,rate)
#     features = fbank_feat #np.hstack((mfcc_feat, d_mfcc_feat, fbank_feat))
#     clusteredFeatures = performClustering(features)
#     print(features.shape)
#     print(clusteredFeatures.shape)
#     return features

# def performClustering(array):
#     arr = np.array(array).astype(float)
#     centroid, _ = kmeans(arr,20)
#     marked,_ = vq(arr, centroid)
#     clusterMap = {}
#     for i in xrange(len(marked)):
#         if marked[i] in clusterMap:
#             clusterMap[marked[i]].append(arr[i])
#         else:
#             clusterMap[marked[i]] = [arr[i]]
    
#     selectedVectors = []
#     for k in clusterMap.keys():
#         l = clusterMap[k]
#         print(len(l))
#         if len(l) < 8:
#             selectedVectors = selectedVectors + l
#         else:
#             selectedVectors = selectedVectors + [l[i] for i in random.sample(xrange(len(l)), 8)]
#     return np.array(selectedVectors)

def getFiles(directoryName):
    onlyFiles = [f for f in listdir(directoryName) if isfile(join(directoryName, f))]
    return onlyFiles

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8002)))
