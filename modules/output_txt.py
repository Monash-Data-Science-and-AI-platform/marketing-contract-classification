from datetime import datetime


def output_txt(file_path):
   
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")

    output_txt=file_path+'/'+dt_string+'.txt'

    return output_txt


    