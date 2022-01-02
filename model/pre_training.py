import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
import sklearn.metrics as skm
import sys
import json
import wandb
from wandb.keras import WandbCallback

preTraining_param_file=open('marketing-contract-classification/model/modules/pre_Training/preTraining.json')
preTraining=json.load(preTraining_param_file)
sys.path.append(preTraining['modules_path'])
from preTraining_data import PreTraining_data 
from preTraining_dataset import PreTraining_dataset 

wandb.init(project=preTraining['project_name'], entity=preTraining['wandb_entity'])
wandb.config = {
  "learning_rate": preTraining['learning_rate'],
  "epochs": preTraining['epochs'],
  "batch_size": preTraining['batch_size']
}

excel_data=PreTraining_data(preTraining['preTraining_data_path'])
processedText=excel_data.get_data()
process_Dataset=PreTraining_dataset(processedText,preTraining['pre_Training_model_path'],preTraining['max_length'])
train_dataset=process_Dataset.get_dataset()

from transformers import TFBertForMaskedLM
model = TFBertForMaskedLM.from_pretrained(preTraining['pre_Training_model_path'])
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)#input_ids
optimizer = tf.keras.optimizers.Adam(preTraining['learning_rate'])
model.compile(optimizer=optimizer,metrics=['accuracy'],loss=loss) # what loss function should I use???

model.fit(train_dataset.shuffle(preTraining['shuffle']).batch(preTraining['batch_size']),epochs=preTraining['epochs'],batch_size=preTraining['batch_size'],callbacks=[WandbCallback()])

model.save_pretrained(preTraining['save_model_path'])