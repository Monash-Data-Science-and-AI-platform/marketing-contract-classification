import pandas as pd
import numpy as np
import openpyxl

#class for data extractions from excel file containing:
#'processedText': sentences
# and labels
#default labels are ['DC','CA_1','CA_2','R&R','IE','Flex'], with a 'None' class always be appended in the output
class Data_extract:

  def __init__(self,train_paths,val_paths,keys=['DC','CA_1','CA_2','R&R','IE','Flex'],min_length=8):

    #train_paths: list of paths to excel files as training set
    #val_paths: list of paths to excel files as validation set
    #keys: labels to be the output, 'None' is always appended
    #min_length: minimum word count per sentences in 'processedText' that are interested

    self.keys=np.array(keys)
    self.train_paths=np.array(train_paths)
    self.val_paths=np.array(val_paths)
    self.min_length=min_length

  def return_label (self,paths):
    #method for processing and returning the labels
    #return lables as np array
    df=self.read_excel_data(paths)
    
    #break 1's and 2's in CA labels in the excel file
    if any("CA_1" in key for key in self.keys):
      df['CA_1']=0
      df.loc[df['CA']==1,'CA_1']=1

    if any("CA_2" in key for key in self.keys):
      df['CA_2']=0
      df.loc[df['CA']==2,'CA_2']=1

    #append "None" class into the label
    df['None']=1
    None_arr=df['None'].to_numpy()

    #iterrate across all available labels
    for key in self.keys:
      None_arr=np.multiply(None_arr,np.array(1-(df[key].to_numpy())))#for rows with all 0's in the labels provided, "None"=1

    df['None']=None_arr#append the processed 'None' to the dataframe

    temp_keys=np.append(self.keys,'None')
    
    labels=df[temp_keys].to_numpy()
    return labels

  def return_text(self,paths):

    #method for extracting the sentences from excel file
    #return sentences as np.array
    df=self.read_excel_data(paths)
    sentences=df['processedText'].to_numpy()

    return sentences

  def read_excel_data(self,paths):

    
    #method for extracting data from excel file
    #return df as pandas.DataFrame
    df=pd.read_excel(paths[0])#take the f
    paths=np.delete(paths,0)#remove the first path as it is assigned to df already
    
    for path in paths:#iterate across the remaning paths
      
      new=pd.read_excel(path)
      frames=[df,new]
      df=pd.concat(frames)

    df=df.loc[df['WC']>=self.min_length]#only interested with sentences greater/eqaul to min_length

    return df

  def train_text(self):#return train set sentences
    
    return self.return_text(self.train_paths)

  def train_labels(self):#return train set labels

    return self.return_label(self.train_paths)

  def val_text(self):#return validation set sentences

    return self.return_text(self.val_paths)

  def val_labels(self):#return validation set labels

    return self.return_label(self.val_paths)