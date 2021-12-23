import tensorflow as tf
import numpy as np
from transformers import TFBertForSequenceClassification, BertConfig


class Fine_tune_model:

    def __init__(self,model_path,config_path,keys=['DC','CA_1','CA_2','R&R','IE','Flex']):
        
        
        self.model_path=model_path
        self.config_path=config_path
        self.keys=keys
        self.config={}
        

    def update_config(self,keys,config_path):
    
        config = BertConfig.from_pretrained(config_path)

        if ~(np.isin('None',np.array(keys))):
            keys.append('None')
        

        id2label={}
        label2id={}

        counter=0
        
        for key in keys:
            
            current_id2label={str(counter):str(key)}
            current_label2id={str(key):counter}
            id2label.update(current_id2label)
            label2id.update(current_label2id)
            counter+=1

        config.id2label=id2label
        config.label2id=label2id
        config.num_labels=len(keys)

        return config

    def get_model(self):
        self.config = self.update_config(self.keys,self.config_path)
        model = TFBertForSequenceClassification.from_pretrained(self.model_path,config=self.config)

        return model

    def get_config(self):

        return self.config





    

    