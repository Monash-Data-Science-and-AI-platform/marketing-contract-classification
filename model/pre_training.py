import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
import sklearn.metrics as skm
import sys
import json
import wandb
import glob
from wandb.keras import WandbCallback

#open the json file that contains all the path and parameter
preTraining_param_file=open('marketing-contract-classification/model/modules/pre_Training/preTraining.json')
preTraining=json.load(preTraining_param_file)

#open and load all the custom modules
sys.path.append(preTraining['modules_path'])
from preTraining_data import PreTraining_data 
from preTraining_dataset import PreTraining_dataset 

#define the wandb properties
wandb.init(project=preTraining['project_name'], entity=preTraining['wandb_entity'])
wandb.config = {
  "learning_rate": preTraining['learning_rate'],
  "epochs": preTraining['epochs'],
  "batch_size": preTraining['batch_size']
}

#get the path of all csv and xlsx files in the folder directories
csv_file=glob.glob(preTraining['preTraining_folder_path']+"/*.csv")
xlsx_file=glob.glob(preTraining['preTraining_folder_path']+"/*.xlsx")

#obtain the sentences
excel_data=PreTraining_data(csv_file+xlsx_file)
processedText=excel_data.get_data()

#define the class to generate the dataset
process_Dataset=PreTraining_dataset(processedText.astype(str),preTraining['pre_Training_model_path'],preTraining['max_length'])
train_dataset=process_Dataset.get_dataset()#get the dataset

from transformers import TFBertForMaskedLM
model = TFBertForMaskedLM.from_pretrained(preTraining['pre_Training_model_path'])#define the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)#define the loss function
optimizer = tf.keras.optimizers.Adam(preTraining['learning_rate'])#define the optimizer
model.compile(optimizer=optimizer,metrics=['accuracy'],loss=loss) # compile the model

print('training started')
for i in range(preTraining['epochs']):
  ####
  model.fit(train_dataset.shuffle(preTraining['shuffle']).batch(preTraining['batch_size']),epochs=1,batch_size=preTraining['batch_size'],callbacks=[WandbCallback()],verbose=2)#train the model
  
  model.save_pretrained(preTraining['save_model_path']+"/epoch_"+str(i))#save the model

  

