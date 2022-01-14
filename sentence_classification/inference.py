import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
import json
import datetime
from transformers import AutoTokenizer, TFBertForSequenceClassification, BertConfig

#open and load the json file that contains all the path and parameter
param_file=open('marketing-contract-classification/sentence_classification/inference/inference.json')
param=json.load(param_file)

#read the csv or xlsx file
if param['data_file_path'].endswith('.csv'):
  df=pd.read_csv(param['data_file_path'],dtype='string')

elif param['data_file_path'].endswith('.xlsx'):
  df=pd.read_excel(param['data_file_path'],engine='openpyxl',dtype='string')

#extratc the sentence and convert to np array
sentences=df['processedText'].to_numpy()

#take only the defined range of sentences
sentences=sentences[param['inclusive_starting_row']:param['exclusive_ending_row']]

tokenizer=AutoTokenizer.from_pretrained(param['tokenizer_path'])#define the tokenizer
config=BertConfig.from_pretrained(param['config_path'])#load the config
model = TFBertForSequenceClassification.from_pretrained(param['model_path'],config=config)#define the model
inputs = tokenizer(sentences.tolist(), padding='max_length',truncation=True,max_length=512,return_tensors="tf")#obtain the encodings

model.summary()#print the summary of the model

outputs = model(inputs.input_ids,training=False)#get the predictions from the model
logits = outputs.logits#obtain the output logits

scaled_result=tf.math.sigmoid(logits).numpy()#scale the output logits with sigmoid function
standardized_result=np.zeros(scaled_result.shape)#define the np array of the standarduzed result
standardized_result[scaled_result>=0.5]=1#standardized the scaled logits

standardized_result_trans=np.transpose(standardized_result)#transpose the array for outputing the result

new_df=pd.DataFrame()#initialize new pandas dataframe
new_df['input_sentences']=sentences#add the sentences into the dataframe
new_df['doc path']=np.full((len(sentences)),param['data_file_path'])#transfer the document path

for i in range(len(param['keys'])):#transfer all the predictions into the df
  new_df[param['keys'][i]]=standardized_result_trans[i]


new_df.to_csv(param['output_path'])#output the df to a .csv file
