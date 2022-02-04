import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
import json
import datetime
import glob
import sys
from transformers import AutoTokenizer, TFBertForSequenceClassification, BertConfig

#open and load the json file that contains all the path and parameter
param_file=open('marketing-contract-classification/sentence_classification/inference/inference.json')
param=json.load(param_file)

sys.path.append(param['modules_path'])
from extract_data import Extract_data

#get the path of all csv and xlsx files in the folder directories
csv_file=glob.glob(param['dataset_folder_path']+"/*.csv")
xlsx_file=glob.glob(param['dataset_folder_path']+"/*.xlsx")

#obtain the sentences
excel_data=Extract_data(csv_file+xlsx_file)
sentences=(excel_data.get_data()).astype(str)
print('Data extracted\n')
tokenizer=AutoTokenizer.from_pretrained(param['tokenizer_path'])#define the tokenizer
config=BertConfig.from_pretrained(param['config_path'])#load the config
model = TFBertForSequenceClassification.from_pretrained(param['model_path'],config=config)#define the model
print('Predictions started\n')
inputs = tokenizer(sentences.tolist(), padding='max_length',truncation=True,max_length=512,return_tensors="tf")#obtain the encodings

model.summary()#print the summary of the model
print('Prediction done, processing the raw output\n')
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

print('Finalizing...\n')
new_df.to_csv(param['output_path'])#output the df to a .csv file
