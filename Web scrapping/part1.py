
import bs4 as bs
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import os
import time


### PART 1 - Download Metadata ###
driver = webdriver.Chrome(options=Options(), executable_path='chromedriver.exe')
main_url = 'https://www.cards.commerce.state.mn.us/CARDS/'
driver.get(main_url)

area_of_interest = Select(driver.find_element_by_id('form:j_id1979103314_19954511'))
area_of_interest.select_by_visible_text('Franchise Registrations')
time.sleep(10)
document_type = Select(driver.find_element_by_name('form:j_id1252080239_1_19954493:3:j_id1979103314_2_199547ee'))
document_type.select_by_visible_text('Disclosure Form')

driver.find_element_by_id('form:j_id1252080239_1_19954729').click()
time.sleep(10)

metadata = pd.DataFrame(columns=['ID','PDF saved','Document Link','Franchisor','Name of Franchise','Document Type','Year',
                                   'File Number','Notes','Received Date','Added On'])

visible_doc=Select(driver.find_element_by_id('form:tbl:j_id1252080239_1_19954642'))#change to 100 at the top button
visible_doc.select_by_visible_text('100')

time.sleep(10)
next_button_id='form:tbl:j_id1252080239_1_199541d1'#top next button id, top buttons work fine, but not the bottom one
end=0
doc_count = 0
while True:
    soup = bs.BeautifulSoup(driver.page_source)
    table = soup.find('table')
    rows = table.find_all('tr')
    for _r, r in enumerate(rows):
        if _r == 0: continue
        cells = r.find_all('td')

        if (cells[0].text)=='No records found.':#if the result in table is no record found, break the loop
            end=1
            break

        metadata.loc[doc_count] = ['PDF{:04d}'.format(doc_count), '-',
                                     'https://www.cards.commerce.state.mn.us' + cells[0].find('a')['href']
                                     ] + \
                                    [c.text for c in cells[1:]]
        # print(metadata.loc[doc_count])
        doc_count += 1



    
    if (driver.find_element_by_id(next_button_id).is_enabled()):#check the next button is enabled or not
        driver.find_element_by_id(next_button_id).click()#if yes click it and wait for 10s
        time.sleep(10)
    else:
        break#break the loop if the button is disabled
    
    if end:#break the loop if there is no result(most of the time the button is not disabled)
        break
     
    

metadata.to_csv('metadata.csv')



