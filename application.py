#!/usr/bin/env python

import os
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import json
from sklearn.externals import joblib
import numpy as np
import collections
import traceback
import subprocess

from flask import Flask, request

application = Flask(__name__)

emotion_array = ['happy', 'sadness', 'angry']

def convertFile(filename):
    sound = AudioSegment.from_file(filename, "3gpp")
    sound.export("/inputFile.wav", format="wav")
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
                test_audio = getFeature(os.path.join("./", filename))
                print("Preparing file")
                testFeatureList = []
                for a in test_audio:
                    testFeatureList.append(a)
                clf = joblib.load('./data/model/model.pkl')
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
                return json.dumps(jobject)
            except Exception as ex:
                jobject = {}
                jobject['status'] = "Error"
                jobject['Message'] = traceback.format_exc()
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
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    features = np.hstack((mfcc_feat, d_mfcc_feat, fbank_feat))
    return features

def getFiles(directoryName):
    onlyFiles = [f for f in listdir(directoryName) if isfile(join(directoryName, f))]
    return onlyFiles

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5003)))
