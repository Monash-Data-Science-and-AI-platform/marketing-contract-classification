import tensorflow as tf
import numpy as np
from transformers import TFBertForSequenceClassification, BertConfig


class Fine_tune_model:

    def __init__(self,model_path,config_path,keys=['DC','CA_1','CA_2','R&R','IE','Flex']):
        
        self.model_path=model_path#path to the pre-trained bert model
        self.config_path=config_path#path to the config.json of the pre-trained model
        self.keys=keys#keys needed to fine-tune the model
        self.config={}#configurtion of the model
        

    def update_config(self,keys,config_path):
    
        config = BertConfig.from_pretrained(config_path)#load the config file

        if ~(np.isin('None',np.array(keys))):
            keys.append('None')#append 'none' if not included in the keys
        
        #initialize the config parameters
        id2label={}
        label2id={}

        counter=0#counter to count the number of keys
        
        for key in keys:
            
            #create the arg for the parameters
            current_id2label={str(counter):str(key)}
            current_label2id={str(key):counter}
            #append to the dict
            id2label.update(current_id2label)
            label2id.update(current_label2id)
            counter+=1

        #assign the new parameters' arguments
        config.id2label=id2label
        config.label2id=label2id
        config.num_labels=len(keys)

        return config

    def get_model(self):
        self.config = self.update_config(self.keys,self.config_path)#get the new configurations
        model = TFBertForSequenceClassification.from_pretrained(self.model_path,config=self.config)#define the model

        return model

    def get_config(self):#method to get the updated config

        return self.config





    

    