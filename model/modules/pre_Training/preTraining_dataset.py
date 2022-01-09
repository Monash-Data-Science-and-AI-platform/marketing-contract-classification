import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel


class PreTraining_dataset:

    def __init__(self,data,tokenizer_path,max_length):

        self.data=data#data is the sentences
        self.tokenizer_path=tokenizer_path#path to tokenizer(the pre-trained model)
        self.max_length=max_length#max length of the encodings

    def np_to_tensor(self,arg):#function to convert the encodings(list/np) to tf tensor
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        return arg

    def get_dataset(self):#method to get the dataset

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)#define the tokenizer

        encodings = tokenizer(self.data.tolist(), truncation=True, padding='max_length', max_length=512)#convert words into encodings
        
        
        encode=np.array(encodings.input_ids)#convert to np array

        arr=np.random.rand(*encode.shape)#create an array with the same same with the encodings but with random values 0 to 1

        mask_arr = (arr< 0.15)*(encode!=101)*(encode!=102)*(encode!=0)#create a masking array which is 1 when conditions are met
        #101,102,0 are special tokens in bert vocab
        encodings.attention_mask=mask_arr

        input_ids=np.array(encodings.input_ids)#obtain the input ids only from the encodings
        input_ids[mask_arr==True]=103#assign the [MASK] token to input ids based on mask_arr
        encodings.input_ids=input_ids#re-defined the input ids

        train_dataset = tf.data.Dataset.from_tensor_slices((#create the dataset
            dict(encodings),
            self.np_to_tensor(encode)#label
        ))


        return train_dataset



