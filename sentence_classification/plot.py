import matplotlib.pyplot as plt
import json
import numpy as np

#import the parameters
param_file=open('marketing-contract-classification/sentence_classification/modules/plot/plot.json')
param=json.load(param_file)

#load the raw data from json file
file=open(param['raw_data_file_path'])
data=json.load(file)
file.close()

#initialize the variables for plotting
keys=param['keys']
epochs=np.arange(0,len(data))
weighted_average=np.zeros(len(data))
f1_scores=np.zeros((len(keys),len(data)))

#extract the data from the json dict
epoch=0
for key in data:
    counter=0

    for subkey in data[key]:
        if(counter<len(keys)):

            f1_scores[counter][epoch]=data[key][subkey]['f1-score']
            counter+=1

    weighted_average[int(key)]=data[key]['weighted avg']['f1-score']
    epoch+=1


#plot the f1-scores for each keys
for i in range(len(keys)):
  plt.plot(epochs, f1_scores[i],label=keys[i])

#plotting the weighted average
plt.plot(epochs,weighted_average,label='weighted av')

#naming the x-axis
plt.xlabel(param['xlabel'])
# naming the y axis
plt.ylabel(param['ylabel'])
# giving a title to my graph
plt.title(param['title'])#add the pre-training epoch
plt.legend(bbox_to_anchor=(1.5,0.5),loc='center right')
# function to show the plot
plt.show()

#export the graph
plt.savefig(param['save_figure_path'],bbox_inches="tight")
