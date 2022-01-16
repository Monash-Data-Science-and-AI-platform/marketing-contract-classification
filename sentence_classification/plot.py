import matplotlib.pyplot as plt
import json
import numpy as np

file=open('ec18_scratch/ilee0022/output_result/pt14_raw.json')
data=json.load(file)
file.close()
keys=[
        "DC","CA_1","CA_2","R&R","IE","Flex","None"
]

epochs=np.arange(0,len(data))
weighted_average=np.zeros(len(data))
f1_scores=np.zeros((len(keys),len(data)))

epoch=0
for key in data:
    counter=0

    for subkey in data[key]:
        if(counter<len(keys)):

            f1_scores[counter][epoch]=data[key][subkey]['f1-score']
            counter+=1

    weighted_average[int(key)]=data[key]['weighted avg']['f1-score']
    epoch+=1



for i in range(len(keys)):
  plt.plot(epochs, f1_scores[i],label=keys[i])

# naming the x axis
plt.plot(epochs,weighted_average,label='weighted av')

plt.xlabel('epoch')
# naming the y axis
plt.ylabel('f1 scores')
# giving a title to my graph
plt.title('F1 scores over epochs')
plt.legend(bbox_to_anchor=(1.04,0.5),loc='center right')
# function to show the plot
plt.show()

plt.savefig('ec18_scratch/ilee0022/pt_plus_ft_logs/pt14.png',bbox_inches="tight")
