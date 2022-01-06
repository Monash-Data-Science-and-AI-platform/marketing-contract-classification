import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel


class PreTraining_dataset:

    def __init__(self,data,tokenizer_path,max_length):

        self.data=data
        self.tokenizer_path=tokenizer_path
        self.max_length=max_length

    def np_to_tensor(self,arg):
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        return arg

    def get_dataset(self):

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        a=np.zeros(len(self.data.tolist()))

        for i in range(len(self.data.tolist())):

            try:
                encodings = tokenizer(self.data.tolist(), truncation=True, padding='max_length', max_length=512)
                a[i]=encodings
            except:
                print(self.data.tolist()[i])
        
        
        encode=np.array(encodings.input_ids)

        arr=np.random.rand(*encode.shape)

        mask_arr = (arr< 0.15)*(encode!=101)*(encode!=102)*(encode!=0)

        encodings.attention_mask=mask_arr

        input_ids=np.array(encodings.input_ids)
        input_ids[mask_arr==True]=103
        encodings.input_ids=input_ids

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            self.np_to_tensor(encode)#label
        ))


        return train_dataset



