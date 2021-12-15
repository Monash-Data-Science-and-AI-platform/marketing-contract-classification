import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertConfig

class Fine_tune_model:

    def __init__(self,model_path,config_path,keys=['DC','CA_1','CA_2','R&R','IE','Flex']):
        
        
        self.model_path=model_path
        self.config_path=config_path
        self.keys=keys
        

    def update_config(self,keys,config_path):
    
        config = BertConfig.from_pretrained(config_path)
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
        config = self.update_config(self.keys,self.config_path)
        model = TFBertForSequenceClassification.from_pretrained(self.model_path,config=config)

        #add a layer, keras sigmoid
        #is it that the higher the accuracy the better the lost function?

        input_ids = tf.keras.Input(shape=(512,),dtype='int32',name='input_ids')#dimension=shape of array 512*batch size,int*float=float

        transformer = model(input_ids)    
        t_logits = transformer.logits # get output_hidden_states
        output = tf.keras.layers.Activation('sigmoid',name='sigmoid')(t_logits)#tf keras.#dont dense
        model = tf.keras.models.Model(inputs = input_ids, outputs = output, name='tf')

        return model







    

    