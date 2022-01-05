import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
import json
import sklearn.metrics as skm
import wandb
from wandb.keras import WandbCallback
import sys
import datetime

param_file=open('marketing-contract-classification/model/modules/fine_tuning/parameter.json')
path_file=open('marketing-contract-classification/model/modules/fine_tuning/path.json')

param=json.load(param_file)
path=json.load(path_file)

sys.path.append(path['modules_path'])
from data_extract import Data_extract
from fine_tune_model import Fine_tune_model 
from prepare_dataset import Prepare_dataset 
from result_process import result_process

wandb.init(project=param['project_name'], entity=param['wandb_entity'])
wandb.config = {
  "learning_rate": param['learning_rate'],
  "epochs": param['epochs'],
  "batch_size": param['train_batch_size']
}

extract=Data_extract(path['training_data_path'],path['validation_data_path'],param['keys'])

train_features=extract.train_text()
train_labels=extract.train_labels()

val_features=extract.val_text()
val_labels=extract.val_labels()

dataset_processor=Prepare_dataset(path['tokenizer_path'],train_features,val_features,train_labels,val_labels)
val_dataset=dataset_processor.get_val_dataset()
train_dataset=dataset_processor.get_train_dataset()

fine_tune_model=Fine_tune_model(path['model_path'],path['config_file_path'],param['keys'])
model=fine_tune_model.get_model()
optimizer = tf.keras.optimizers.Adam(param['learning_rate'])
model.compile(optimizer=optimizer, loss=param['loss_function']) # change to categorical for single-class

with open(path['output_file_path'], "w") as f:
        f.write('%s\n' % datetime.datetime.now())
        f.write("Keys: %s" %param['keys'])
        f.write("\n")
        f.write(" %s" %model.summary(print_fn=lambda x: f.write(x + '\n')))
        f.write("\n")


for i in range(param['epochs']):
  model.fit(train_dataset.shuffle(param['shuffle']).batch(param['train_batch_size']), epochs=1, batch_size=param['train_batch_size'], validation_data=val_dataset.batch(param['val_batch_size']),callbacks=[WandbCallback()],verbose=2)#will it behaves properly?????
  scaled_pred=model.predict(val_dataset.batch(param['val_batch_size']),batch_size=param['val_batch_size'])
  result_process(scaled_pred.logits,val_labels,path['output_file_path'],i)

model.save_pretrained(path['save_model_path'])
config_save=fine_tune_model.get_config()
config_save.save_pretrained(path['save_config_path'])