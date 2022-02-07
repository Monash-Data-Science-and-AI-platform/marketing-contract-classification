import pandas as pd
import numpy as np
import openpyxl


class Extract_data:

    def __init__(self,excel_paths):

      self.excel_paths=excel_paths
      
    def read_excel_data(self,paths):

      #method for extracting data from excel file
      #return df as pandas.DataFrame
      if paths[0].endswith('.csv'):
        df=pd.read_csv(paths[0])

      elif paths[0].endswith('.xlsx'):
        df=pd.read_excel(paths[0],engine='openpyxl')

      paths=np.delete(paths,0)#remove the first path as it is assigned to df already
      
      for path in paths:#iterate across the remaning paths
        
        error=False#intialise error variable
        if path.endswith('.csv'):#for csv file
          try:
              new=pd.read_csv(path)#read the file
          except:
              error=True#if fails to read the file
              print(path)#print the file's path
              pass

        elif path.endswith('.xlsx'):#for xlsx file
          new=pd.read_excel(path,engine='openpyxl')#read the file
        
        if(error==False):#if  no error occured
            new=new.dropna()#remove all empty rows
            frames=[df,new]
            df=pd.concat(frames)#concate the current df with the newly read df

      
      return df

    def return_text(self,paths):

      #method for extracting the sentences from excel file
      #return sentences as np.array
      df=self.read_excel_data(paths)
      sentences=df['processedText'].to_numpy()

      return sentences
    def get_source_doc(self):

      df=self.read_excel_data(self.excel_paths)
      source_doc=df['contractName'].to_numpy()

      return source_doc

    def get_data(self):#return train set sentences
    
      return self.return_text(self.excel_paths)