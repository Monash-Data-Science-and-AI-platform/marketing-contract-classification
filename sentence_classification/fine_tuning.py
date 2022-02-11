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
import os
import tensorflow_addons as tfa

print("Code started to run\n")
#open the json file that contains all the path and parameters
param_file=open('marketing-contract-classification/sentence_classification/modules/fine_tuning/parameter.json')
path_file=open('marketing-contract-classification/sentence_classification/modules/fine_tuning/path.json')

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
dataset_processor=Prepare_dataset(path['tokenizer_path'],train_features,val_features,train_labels,val_labels,param['keys'])
val_dataset=dataset_processor.get_val_dataset()#get the validation dataset
train_dataset=dataset_processor.get_train_dataset()#get the training dataset
train_dataset_summary=dataset_processor.get_train_dataset_summary()
val_dataset_summary=dataset_processor.get_val_dataset_summary()

#set the class weight based on the parameter.json
if (param['weighted']==0):
  class_weights=None#if 0, set the class weight as none(balanced class weight)
else:
  class_weights=dataset_processor.get_class_weight()#else, generate class weight as inversely proportional to the class size

fine_tune_model=Fine_tune_model(path['model_path'],path['config_file_path'],param['keys'])#define class that handles the initialization of model
model=fine_tune_model.get_model()#get the model
optimizer = tf.keras.optimizers.Adam(param['learning_rate'])#define the optimizer

if (param['loss_function'])=='sigmoid_focal_crossentropy':
  loss=tfa.losses.SigmoidFocalCrossEntropy()
else:
  loss=param['loss_function']

model.compile(optimizer=optimizer, loss=loss) #compile the model


now = datetime.datetime.now()#get the time and date
output_folder_path=path['output_folder_path']+"/"+param['project_name']+"/"+str(now.strftime("%d.%m.%Y %H:%M:%S"))#create the output folder directory path in string
os.makedirs(output_folder_path,exist_ok=True)#create the output folder directory


with open(output_folder_path+'/result.txt', "w") as f:#output the model's summary
  f.write('%s\n' %param['project_name'])
  f.write('%s\n' % now.strftime("%d/%m/%Y %H:%M:%S"))#output the date and time
  f.write("Keys: %s\n" %param['keys'])#output the keys
  f.write("Training class size %s\n" % str(train_dataset_summary))
  f.write("Validation class size: %s\n" % str(val_dataset_summary))
  f.write("Learning rate: %s\n" %param['learning_rate'])#output the keys
  f.write(" %s\n" %model.summary(print_fn=lambda x: f.write(x + '\n')))#print the summary of the model
  f.close()

#initialize variables for plotting and outputting raw datas
epochs=np.arange(0,param['epochs'])#contain integer from 0 to final epoch
f1_scores=np.zeros((len(param['keys']),param['epochs']))#vector to store f1-scores for each label in each epoch
weighted_average=np.zeros(param['epochs'])#vector containing weighted average for each epoch
output_dict={}#dict containg raw data from sklearn.metrics.classification_report

print('training started')
#training
for i in range(param['epochs']):
  model.fit(train_dataset.shuffle(param['shuffle']).batch(param['train_batch_size']), epochs=1, batch_size=param['train_batch_size'],class_weight=class_weights, validation_data=val_dataset.batch(param['val_batch_size']),callbacks=[WandbCallback()],verbose=2)#train the model
  pred=model.predict(val_dataset.batch(param['val_batch_size']),batch_size=param['val_batch_size'])#get the predictions
  report_dict=result_process(pred.logits,val_labels,output_folder_path+'/result.txt',i)#class to process the result
  
  output_dict[str(i)]=report_dict#obtain the skm.classification report
  
  weighted_average[i]=report_dict['weighted avg']['f1-score']#get weighted av for plotting
  counter=0

  #extract f1-score for each keys
  for key in report_dict:
    f1_scores[counter][i]=report_dict[key]['f1-score']
    counter+=1

    if counter==len(param['keys']):#stop the loop when the counter is at the last label, other element in the dict are not label
      break
  #output the raw data
  with open(output_folder_path+'/raw_data.json','w') as f:
    json.dump(output_dict, f, indent = 4)
    f.close()

  
  model.save_pretrained(path['save_model_path']+"/epoch_"+str(i))#save the model
  config_save=fine_tune_model.get_config()#get the updated config file
  config_save.save_pretrained(path['save_model_path']+"/epoch_"+str(i))#save the config file}




#plot the f1-score for each keys
for i in range(len(f1_scores)):
  plt.plot(epochs, f1_scores[i],label=param['keys'][i])

#plot the weighted average
plt.plot(epochs,weighted_average,label='weighted average')
# naming the x axis
plt.xlabel(param['xlabel'])
# naming the y axis
plt.ylabel(param['ylabel'])
#changing the resolution of both axes
plt.xticks(np.arange(0,101, 5.0)) 
plt.yticks(np.arange(0,1.01, 0.05)) 
#add grid to the plot
plt.grid()
# giving a title to my graph
plt.title(param['plot_title'])
plt.legend(bbox_to_anchor=(1.5,0.5),loc='center right')
# function to show the plot
plt.show()
#export the graph
plt.savefig(output_folder_path+'/result_graph.png',bbox_inches="tight")