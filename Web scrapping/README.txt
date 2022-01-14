Summary:
This folder contains 
-download_pdf.py: To download pdf based on data in metadata.csv
-info_obtain.py: obtain information from 'https://www.cards.commerce.state.mn.us/CARDS/'
-metadata.csv: output excel file for info_obtain.py, it contains all the information obatianed from 'https://www.cards.commerce.state.mn.us/CARDS/'

___________________________________________________________________________________________________________________________________________________
Preparation:

Before running any script in this folder , 
1. Install chromedriver.exe in this folder, it can be downloaded from: https://chromedriver.chromium.org/downloads
2. Make sure the local computer/conda environment is installed with
	-pandas
	-PyPDF2
	-bs4
	-selenium

___________________________________________________________________________________________________________________________________________________
Running the script:
-simply run info_obtain.py first
-then, inspect metadata.csv to make sure the outputs are normal
-run download_pdf.py
-manually download the pdf that are marked as 'corrupted' in metadata.csv based on the url in metadata.csv if needed
