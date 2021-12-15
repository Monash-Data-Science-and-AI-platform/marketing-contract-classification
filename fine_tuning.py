! pip install transformers datasets
! pip install wandb
! wandb login

import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
import string 
import json
import sklearn.metrics as skm
import wandb
from wandb.keras import WandbCallback

import sys

param_f=open('/content/drive/MyDrive/legal_bert/modules/parameter.json')
path_f=open('/content/drive/MyDrive/legal_bert/modules/path.json')

param=json.load(param_f)
path=json.load(path_f)

sys.path.append(path['modules_path'])
from data_extract import Data_extract as DE
from fine_tune_model import Fine_tune_model as FTM
from prepare_dataset import Prepare_dataset as PD
from result_process import result_process
from output_txt import output_txt



wandb.init(project="legal_bert_trial", entity="ilee0022")

extract=DE(path['training_path'],path['validation_path'],param['keys'])

train_f=extract.train_text()
train_labels=extract.train_labels()

val_f=extract.val_text()
val_labels=extract.val_labels()

dataset_processor=PD(path['tokenizer_path'],train_f,val_f,train_labels,val_labels)
val_dataset=dataset_processor.get_val_dataset()
train_dataset=dataset_processor.get_train_dataset()

ftm=FTM(path['model_path'],path['config_file_path'],param['keys'])
model=ftm.get_model()
optimizer = tf.keras.optimizers.Adam(param['learning_rate'])
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)#tried Binary,Sparse and Category #refer to each class
print(model.summary())
model.compile(optimizer=optimizer, loss=loss) # can also use any keras loss fn

txt_path=output_txt(path['output_file_path'])

with open(txt_path, "w") as f:
        f.write("Keys: %s" %param['keys'])
        f.write("\n")

#callback early function
for i in range(param['epochs']):
  model.fit(train_dataset.shuffle(param['shuffle']).batch(param['train_batch_size']), epochs=param['epochs'], batch_size=param['train_batch_size'], validation_data=val_dataset.batch(param['val_batch_size']),callbacks=[WandbCallback()])#will it behaves properly?????
  
  scaled_pred=model.predict(val_dataset.batch(32),batch_size=32)#
  result_process(scaled_pred,val_labels,txt_path,i)
  