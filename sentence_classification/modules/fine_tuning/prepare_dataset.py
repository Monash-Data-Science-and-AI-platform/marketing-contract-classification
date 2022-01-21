from transformers import AutoTokenizer, AutoModel
import tensorflow as tf
import numpy as np

class Prepare_dataset:

    def __init__(self,tokenizer_path,train_features,val_features,train_labels,val_labels):

        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)#define the tokenizer
        self.train_features=train_features#train features are the sentences for training
        self.val_features=val_features#val features are the sentences for validation
        self.train_labels=train_labels#lables for the training sentences
        self.val_labels=val_labels#labels for validation sentences

    def prepare_dataset(self,features,labels):

        encodings = self.tokenizer(features.tolist(), truncation=True, padding='max_length', max_length=512)#encode the words to integers

        dataset = tf.data.Dataset.from_tensor_slices((#create the dataset
            dict(encodings),
            self.np_to_tensor(labels)
        ))

        return dataset


    def np_to_tensor(self,arg):#method to convert list/nparray to tf tensor
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        return arg

    def get_train_dataset(self):#method to get the train dataset

        return self.prepare_dataset(self.train_features,self.train_labels)

    def get_val_dataset(self):# method to get validation dataset

        return self.prepare_dataset(self.val_features,self.val_labels)

    def get_class_weight(self):#method to get the class weight which is inversely proportional to the class size
        train_labels_trans=np.transpose(self.train_labels)#transpose the matrix

        total_labels=np.sum(self.train_labels)#number of 1's in the matrix
        unscaled_weight=1/(np.sum(train_labels_trans,axis=1)/total_labels)#get the unscled weight which is inversely proportional to the class size
        scaled_weight=unscaled_weight/unscaled_weight.min()#scale the weight
        normalized_weight=scaled_weight/scaled_weight.sum()#normalize the weight to be sum to 1
        class_weights={}#intialize the python dict
        for i in range(len(normalized_weight)):
            class_weights[i]=normalized_weight[i]#assign the value in the vector into the dict

        return class_weights


