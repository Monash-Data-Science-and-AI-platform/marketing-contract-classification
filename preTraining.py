! pip install transformers datasets

import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
import string 
import re
import sklearn.metrics as skm
import sys
import json

param_f=open('/content/drive/MyDrive/legal_bert/modules/parameter.json')
path_f=open('/content/drive/MyDrive/legal_bert/modules/path.json')
preTrain_f=open('/content/drive/MyDrive/legal_bert/modules/preTraining.json')

path=json.load(path_f)
preTraining=json.load(preTrain_f)

sys.path.append(path['modules_path'])
from preTraining_data import PreTraining_data as PTD
from preTraining_dataset import PreTraining_dataset as PTDS

excel_data=PTD(preTraining['preTraining_data_path'])

processedText=excel_data.get_data()

process_Dataset=PTDS(processedText,preTraining['pre_Training_model_path'],preTraining['max_length'])

train_dataset=process_Dataset.get_dataset()


from transformers import TFBertForMaskedLM
model = TFBertForMaskedLM.from_pretrained(preTraining['pre_Training_model_path'])
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)#input_ids
optimizer = tf.keras.optimizers.Adam(preTraining['learning_rate'])


model.compile(optimizer=optimizer,metrics=['accuracy'],loss=loss) # what loss function should I use???

model.fit(train_dataset.shuffle(preTraining['shuffle']).batch(preTraining['batch_size']),epochs=preTraining['epochs'],batch_size=preTraining['batch_size'])#why using sparse works lol

model.save_pretrained(preTraining['save_model_path'])