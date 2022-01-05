import pandas as pd
import numpy as np
import openpyxl


class PreTraining_data:

    def __init__(self,excel_paths):

      self.excel_paths=excel_paths
      
    def read_excel_data(self,paths):

      #method for extracting data from excel file
      #return df as pandas.DataFrame
      if paths[0].endswith('.csv'):
        df=pd.read_csv(paths[0],dtype='string',encoding='Latin-1')

      elif paths[0].endswith('.xlsx'):
        df=pd.read_excel(paths[0],engine='openpyxl')

      paths=np.delete(paths,0)#remove the first path as it is assigned to df already
      
      for path in paths:#iterate across the remaning paths
        
        if path.endswith('.csv'):
          new=pd.read_csv(path,dtype='string',encoding='Latin-1')

        elif path.endswith('.xlsx'):
          new=pd.read_excel(path,engine='openpyxl')
        
        frames=[df,new]
        df=pd.concat(frames)

      return df

    def return_text(self,paths):

      #method for extracting the sentences from excel file
      #return sentences as np.array
      df=self.read_excel_data(paths)
      sentences=df['processedText'].to_numpy()

      return sentences

    def get_data(self):#return train set sentences
    
      return self.return_text(self.excel_paths)
