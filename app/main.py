# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:07:51 2020
@author: win10
"""
import json
import os

# pip install fastapi uvicorn pickle

# 1. Library imports
import uvicorn  ##ASGI
from fastapi import FastAPI
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import librosa
import requests
from pydantic import BaseModel
import tensorflow as tf

"image alpine ou debian et jexecute le script dans la debian du docker \
je fais la docker file from debian je copie le script, jle mets dans debian et je lui dit dexecuter le script"

class Vocal(BaseModel):
    features: list[list[float]]


# Create the app object
app = FastAPI(title="JPWAV-AI prediction REST API ")

print(os.getcwd())

with open('mlp_classifier.pkl', 'rb') as pickle_in:
    model = pickle.load(pickle_in)


# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'JPWAV-API': '/predict for prediction'}


@app.post('/predict')
def predict_sentiment(data: Vocal):
    le = LabelEncoder()
    # print(list((data.features.values())))
    predicted_vector = model.predict(list(data.features))
    predicted_proba = predicted_vector[0]
    # for i in range(len(predicted_proba)):
    #     category = le.inverse_transform(np.array([i]))
    #     print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))
    # for i in range(len(predicted_proba)):
    #     print(predicted_proba[i])
    #     category = le.inverse_transform(np.array([i]))
    #     print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))

    # for i in range(len(predicted_proba)):
    #     category = le.inverse_transform(np.array([i]))
    #     print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))
    # create a dictionary mapping each category to its predicted probability
    ## predictions = {le.inverse_transform(np.array([i]))[0]: predicted_proba[i] for i in range(len(predicted_proba))}

    # return predictions

    return {"Predictions": predicted_vector.tolist()}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# # Load your audio data and compute the MFCC features
#
# audio_data, sample_rate = librosa.load('../Data-Science/Audio_data/our_audios/wass_calme_1.wav',
# res_type='kaiser_fast') mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=40)
#
# # Create a Vocal object with the mfccs data and pass it to the predict_sentiment endpoint
# vocal_data = Vocal(features=mfccs)
# response = requests.post('http://127.0.0.1:8000/predict', json=json.dumps(vocal_data.dict()))
# print(response.json())
# uvicorn main:app --reload
