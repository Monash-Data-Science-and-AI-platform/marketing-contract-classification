### PART 2 - Download PDFs ###

from os.path import exists
import pandas as pd
import requests

metadata = pd.read_csv('metadata.csv',index_col=0)
# TODO loop through each row
# check notes for key terms, check if already saved
# if key terms found, download pdf with ID as name
# mark PDF saved with "Saved"
# Write script that is robust to connection failure / able to continue where you left off
# i.e. check document downloaded before trying to download

for rows in metadata.iterrows():

    print(rows['PDF saved'])
    if(rows['PDF saved']=='-'):

        url = rows['Document Link']
        response = requests.get(url)
        save_path='/data/'+rows['ID']
        with open(save_path, 'wb') as f:
            f.write(response.content)

        if exists(save_path):
            rows['PDF saved']='Saved'
        
        else:
            rows['PDF saved']='Failed to download'


# time.sleep(20)
