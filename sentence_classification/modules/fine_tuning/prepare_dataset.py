from typing import KeysView
from transformers import AutoTokenizer, AutoModel
import tensorflow as tf
import numpy as np

class Prepare_dataset:

    def __init__(self,tokenizer_path,train_features,val_features,train_labels,val_labels,keys):

        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)#define the tokenizer
        self.train_features=train_features#train features are the sentences for training
        self.val_features=val_features#val features are the sentences for validation
        self.train_labels=train_labels#lables for the training sentences
        self.val_labels=val_labels#labels for validation sentences
        self.keys=keys#name of each label

    def prepare_dataset(self,features,labels):

        encodings = self.tokenizer(features.tolist(), truncation=True, padding='max_length', max_length=512)#encode the words to integers

        dataset = tf.data.Dataset.from_tensor_slices((#create the dataset
            dict(encodings),
            self.np_to_tensor(labels)
        ))

        return dataset

    def label_summary(self,labels):
        labels_count=np.sum(labels,axis=0)
        label_summary={}
        keys=self.keys+['None']

        for i in range(len(labels_count)):
            label_summary[keys[i]]=labels_count[i]

        return label_summary

    def get_train_dataset_summary(self):

        return self.label_summary(self.train_labels)

    def get_val_dataset_summary(self):

        return self.label_summary(self.val_labels)

    def np_to_tensor(self,arg):#method to convert list/nparray to tf tensor
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        return arg

    def get_train_dataset(self):#method to get the train dataset

        return self.prepare_dataset(self.train_features,self.train_labels)

    def get_val_dataset(self):# method to get validation dataset

        return self.prepare_dataset(self.val_features,self.val_labels)

    def get_class_weight(self):#method to get the class weight which is inversely proportional to the class size

        total_labels=np.sum(self.train_labels)#number of 1's in the matrix
        unscaled_weight=1/(np.sum(self.train_labels,axis=0)/total_labels)#get the unscled weight which is inversely proportional to the class size
        normalized_weight=unscaled_weight/unscaled_weight.sum()#normalize the weight to be sum to 1
        class_weights={}#intialize the python dict
        for i in range(len(normalized_weight)):
            class_weights[i]=normalized_weight[i]#assign the value in the vector into the dict

        return class_weights


