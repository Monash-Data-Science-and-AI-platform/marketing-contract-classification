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
import matplotlib.pyplot as plt

#open the json file that contains all the path and parameters
param_file=open('marketing-contract-classification/model/modules/fine_tuning/parameter.json')
path_file=open('marketing-contract-classification/model/modules/fine_tuning/path.json')

#load the data from json file
param=json.load(param_file)
path=json.load(path_file)

#get the path of the custom modules
sys.path.append(path['modules_path'])

#import all the custom modules
from data_extract import Data_extract
from fine_tune_model import Fine_tune_model 
from prepare_dataset import Prepare_dataset 
from result_process import result_process

#define the wandb properties
wandb.init(project=param['project_name'], entity=param['wandb_entity'])
wandb.config = {
  "learning_rate": param['learning_rate'],
  "epochs": param['epochs'],
  "batch_size": param['train_batch_size']
}

#extract the data from the files defined
extract=Data_extract(path['training_data_path'],path['validation_data_path'],param['keys'])

train_features=extract.train_text()#sentences for training
train_labels=extract.train_labels()#label for training sentences

val_features=extract.val_text()#sentences for validation
val_labels=extract.val_labels()#labels for validation sentences

#define a class object that handles the processing of sentences and label to dataset
dataset_processor=Prepare_dataset(path['tokenizer_path'],train_features,val_features,train_labels,val_labels)
val_dataset=dataset_processor.get_val_dataset()#get the validation dataset
train_dataset=dataset_processor.get_train_dataset()#get the training dataset

fine_tune_model=Fine_tune_model(path['model_path'],path['config_file_path'],param['keys'])#define class that handles the initialization of model
model=fine_tune_model.get_model()#get the model
optimizer = tf.keras.optimizers.Adam(param['learning_rate'])#define the optimizer
model.compile(optimizer=optimizer, loss=param['loss_function']) #compile the model

with open(path['output_file_path'], "w") as f:#output the model's summary
        f.write('%s\n' % datetime.datetime.now())
        f.write("Keys: %s" %param['keys'])
        f.write("\n")
        f.write(" %s" %model.summary(print_fn=lambda x: f.write(x + '\n')))
        f.write("\n")

epochs=np.arange(0,param['epochs'])
f1_scores=np.zeros((len(param['keys']),param['epochs']))

print('training started')
for i in range(param['epochs']):
  model.fit(train_dataset.shuffle(param['shuffle']).batch(param['train_batch_size']), epochs=1, batch_size=param['train_batch_size'], validation_data=val_dataset.batch(param['val_batch_size']),callbacks=[WandbCallback()],verbose=2)#train the model
  pred=model.predict(val_dataset.batch(param['val_batch_size']),batch_size=param['val_batch_size'])#get the predictions
  report_dict=result_process(pred.logits,val_labels,path['output_file_path'],i)#class to process the result

  counter=0
  for key in report_dict:
    f1_scores[counter][i]=report_dict[key]['f1-score']
    counter+=1

    if counter==len(param['keys']):
      break

  model.save_pretrained(path['save_model_path']+"/epoch_"+str(i))#save the model
  config_save=fine_tune_model.get_config()#get the updated config file
  config_save.save_pretrained(path['save_config_path']+"/epoch_"+str(i))#save the config file

for i in range(len(f1_scores)):
  plt.plot(epochs, f1_scores[i])
# naming the x axis
plt.xlabel('epoch')
# naming the y axis
plt.ylabel('f1 scores')
# giving a title to my graph
plt.title('F1 scores over epochs')
# function to show the plot
plt.show()
plt.savefig('ec18_scratch/ilee0022/pt_plus_ft_logs/pt8_f1.png')