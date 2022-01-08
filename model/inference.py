import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
import json
import datetime
from transformers import AutoTokenizer, TFBertForSequenceClassification, BertConfig

param_file=open('marketing-contract-classification/model/inference/inference.json')
param=json.load(param_file)

if param['data_file_path'].endswith('.csv'):
  df=pd.read_csv(param['data_file_path'],dtype='string')

elif param['data_file_path'].endswith('.xlsx'):
  df=pd.read_excel(param['data_file_path'],engine='openpyxl',dtype='string')

sentences=df['processedText'].to_numpy()
sentences=sentences[param['inclusive_starting_row']:param['exclusive_ending_row']]

tokenizer=AutoTokenizer.from_pretrained(param['tokenizer_path'])
config=BertConfig.from_pretrained(param['config_path'])
model = TFBertForSequenceClassification.from_pretrained(param['model_path'],config=config)
inputs = tokenizer(sentences.tolist(), padding='max_length',truncation=True,max_length=512,return_tensors="tf")

model.summary()

outputs = model(inputs.input_ids,training=False)
logits = outputs.logits

scaled_result=tf.math.sigmoid(logits).numpy()
standardized_result=np.zeros(scaled_result.shape)
standardized_result[scaled_result>=0.5]=1

sentence_index=np.transpose(np.arange(len(standardized_result)))
output=np.hstack((np.split(np.transpose(sentence_index),len(sentence_index)),standardized_result))

with open(param['output_path'], "a") as f:
  f.write('%s\n' % datetime.datetime.now())
  f.write("Prediction:\n")
  f.write('No.\t'+"\t".join(param['keys'])+"\tNone\n")
  np.savetxt(f,output,delimiter='\t',fmt='%d')