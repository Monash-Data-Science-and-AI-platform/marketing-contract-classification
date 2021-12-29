from os.path import exists
import pandas as pd
import urllib.request
import PyPDF2

metadata = pd.read_csv('metadata.csv',index_col=0,keep_default_na=False)

from os.path import exists
import pandas as pd
import urllib.request
import PyPDF2

metadata = pd.read_csv('metadata.csv',index_col=0,keep_default_na=False)

def checkNote(note):#function for checking the notes from excel file

    note=note.lower()#change to lowercase

    if(note==''):#if empty note, return True
        return True
    elif('disclosure document' in note):#if containing 'disclosure document' , then true
        return True
    elif('fdd' in note):#if containing 'fdd', return true
        return True

    else:#else return false
        return False
        
#function to check whether the downloade file is corrupted or not
def checkFile(fullfile):
    with open(fullfile, 'rb') as f:#open the file
        try:
            pdf = PyPDF2.PdfFileReader(f)#read the file
            info = pdf.getDocumentInfo()#get the PDF info of the file
            if info:#if yes, return true, else return False(including errors)
                return True
            else:
                return False
        except:
            return False
        
# TODO loop through each row
# check notes for key terms, check if already saved
# if key terms found, download pdf with ID as name
# mark PDF saved with "Saved"
# Write script that is robust to connection failure / able to continue where you left off
# i.e. check document downloaded before trying to download
PDF_saved=metadata['PDF saved'].to_numpy()#extract the 'PDF saved' into np array to prevent weird pd indexing issue

for rows in range(len(metadata.index)):#iterate across the dataframe rows

    
    if((PDF_saved[rows]=='-')&(checkNote(metadata['Notes'][rows]))):#if the 'PDF saved' is - and the note indicates it is relevant

        url = metadata['Document Link'][rows]#get the url
        response = urllib.request.urlopen(url)#open the url   
        save_path='data/'+metadata['ID'][rows]+'.pdf'#generate the PDF saving path

        with open(save_path, 'wb+') as f:#save the PDF
            f.write(response.read())
            f.close()

        #check the downloaded PDF is corrupted or not
        if checkFile(save_path):#if not
            
            PDF_saved[rows]='Saved'#set status as 'Saved'
            metadata['PDF saved']=PDF_saved
            metadata.to_csv('metadata.csv')#save to excel
        else:
            PDF_saved[rows]='Failed to downoad'#else set as 'Failed to download'
            metadata['PDF saved']=PDF_saved
            metadata.to_csv('metadata.csv')#save to excel

    elif (checkNote(metadata['Notes'][rows])==False):#if the note indicates the link is irrelevant:
        PDF_saved[rows]='Irrelevant'#set status as Irrelevant
        metadata['PDF saved']=PDF_saved
        metadata.to_csv('metadata.csv')#save to excel
