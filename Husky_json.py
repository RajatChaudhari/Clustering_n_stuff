'''
Converting each job profile from husky(docx) data to json.
Processing raw text to clean json format to be used for nlp processing.
'''
from bs4 import BeautifulSoup
from pandas import DataFrame, concat
import mammoth
from os import listdir, walk
from os.path import join as join_path
from glob import glob
import family_classifier
import json
import datetime, os
import Dump_to_json_store
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from pandas import DataFrame, ExcelFile, read_csv, read_excel
import json,os
import pandas as pd


Primary_Job_info = [{'Operating equipment:': None,
   'Driving': None,
   'Shift work': None,
   'On-call': None,
   'Sedentary/Sitting': None,
   'Bending/Crouching': None,
   'Walking': None,
   'Climbing': None,
   'Kneeling/Crawling': None,
   'Keyboarding': None,
   'Travel': None}]
Frequently_Used_Tools = [{'Manual tools': None,
   'Vibration tools': None,
   'Shop tools': None,
   'Power tools': None,
   'Tool belt worn': None,
   'Office equipment': None,
   'Pneumatic tools': None,
   'Welding': None}]
Env_Cond = [{'Noise': None,
   'Outdoors': None,
   'Confined spaces': None,
   ' Extreme heat/cold': None,
   'Chemicals': None,
   'Toxic gases': None,
   'Heights': None,
   'Moving equipment': None,
   'Rotating equipment': None,
   'Uneven surfaces': None,
   'Night time': None,
   'Wet or damp': None}]
Phys_Dem = [{'Sedentary 0-10 pounds': None,
   'Light 11-20 pounds': None,
   'Medium 21-50 pounds': None}]
Phys_req = [{'Lifting': None,
   'Carrying': None,
   'Pushing/pulling': None,
   'Standing:': None}]
all_={'Primary Job Information & Mobility': Primary_Job_info,
    'Frequently Used Tools' : Frequently_Used_Tools,
    'Environmental Conditions': Env_Cond,
    'Physical Demands': Phys_Dem,
    'Physical Requirements': Phys_req
}

clf = family_classifier.family_classify()

class husky_data_json():
    """Processing Taleo data into json format"""
    def __init__(self):
        self.json = {}
        #self.process = process
    def tbl1_to_df(self,html):
        tbl1_dict = dict()
        for row in html.find_all('table')[1].find_all('tr'):
            cells = row.find_all('td')
            key = ' '.join([p.text for p in cells[0].find_all('p')])
            del cells[0]
            value = ' '.join([p.text for p in [cell.find_all('p') for cell in cells][0]]).replace('\n', '').replace('\t', '')
            tbl1_dict[key.strip()] = value
        return DataFrame(tbl1_dict, index=[0])

    def tbl2_to_df(self,html):
        """Fetch responsibilities attributes"""
        tbl_2_dict = dict()
        rows = html.find_all('table')[2].find_all('tr')
        responsabilities = []
        for index in range(0, len(rows)):
            row = rows[index]
            cells = row.find_all('td')
            if len(cells) > 1:
                if index == 0:
                    key = cells[0].text
                    value = cells[1].text
                    tbl_2_dict[key] = value
                elif index != 1:
                    value = cells[0].text
                    responsabilities.append(value)
        responsabilities_str = '$$$%%%&&&'.join(responsabilities)
        tbl_2_dict['responsibilities'] = responsabilities_str
        return DataFrame(tbl_2_dict, index = [0])

    def tbl9_to_df(self, html):
        """Getting competentices attributes"""
        tbl_9_dict = dict()
        rows = html.find_all('table')[9].find_all('tr')
        competency_group = rows[1].text.replace('Select Competency Group: ', '')
        competencies = []
        for index in range(2, len(rows)):
            row = rows[index]
            cells = row.find_all('li')
            for cell in cells:
                competencies.append(cell.text)
        competencies_str = '$$$%%%&&&'.join(competencies)
        tbl_9_dict['competencies'] = competencies_str
        tbl_9_dict['competency_group'] = competency_group
        return tbl_9_dict

    def tbls_reqd_assets_df(self,html, table_index, type_of_qualifications):
        tbl_dict = dict()
        header = html.find_all('table')[table_index].find('th').text
        qualifications = []
        for row in html.find_all('table')[table_index].find_all('tr'):
            cells = row.find_all('td')
            if cells and len(cells) == 3:
                key = cells[0].text
                if key:
                    qualifications.append(key)
        tbl_dict[type_of_qualifications] = '$$$%%%&&&'.join(qualifications)
        return DataFrame(tbl_dict, index=[0])

    def only_reqd_df(self,html, table_index, type_of_qualifications):
        """get required attribute"""
        tbl_dict = dict()
        header = html.find_all('table')[table_index].find('th').text
        qualifications = []
        for row in html.find_all('table')[table_index].find_all('tr'):
            cells = row.find_all('td')
            if cells and len(cells) == 3:
                key = cells[0].text
                reqd = cells[1].text
                if reqd and key:
                    qualifications.append(key)
        tbl_dict[type_of_qualifications] = '$$$%%%&&&'.join(qualifications)
        return DataFrame(tbl_dict, index=[0])

    def only_asset_df(self,html, table_index, type_of_qualifications):
        """Get asset attribute"""
        tbl_dict = dict()
        header = html.find_all('table')[table_index].find('th').text
        qualifications = []
        for row in html.find_all('table')[table_index].find_all('tr'):
            cells = row.find_all('td')
            if cells and len(cells) == 3:
                key = cells[0].text
                asset = cells[2].text
                if asset and key:
                    qualifications.append(key)
        tbl_dict[type_of_qualifications] = '$$$%%%&&&'.join(qualifications)
        return DataFrame(tbl_dict, index=[0])

    def tbl_opts_df(self,html, table_index):
        tbl_dict = dict()
        for row in html.find_all('table')[table_index].find_all('tr'):
            cells = row.find_all('td')
            if cells:
                i = 0
                while(i < len(cells)):
                    try:
                        key = cells[i].text
                        i+=1
                        if key:
                            value = cells[i].text
                            value = value if value != '<Select>' else None
                            tbl_dict[key] = value
                        i+=1
                    except:
                        pass
        return DataFrame(tbl_dict, index=[0])

    def html_to_df(self, html_str, tb=0):
        """Convert html data to dataframe and then to
        json format"""
        html = BeautifulSoup(html_str, 'html.parser')
        if len(html.find_all('table')) == 17:
            if tb==9:
                json_data =  self.tbl9_to_df(html)
                for key, value in json_data.items():
                    if '$$$%%%&&&' in value:
                        json_data[key] = json_data[key].strip().split('$$$%%%&&&')
                    else:
                        json_data[key] =  json_data[key].strip()
                json_data_ = {}
                for key, value in json_data.items():
                    self.json[key.strip().replace(':','')] = json_data[key]

            tbl1 = self.tbl1_to_df(html)
            tbl2 = self.tbl2_to_df(html)
            all_tbls = [tbl1, tbl2]
            tbl=concat(all_tbls, axis = 1, join='inner')
            json_str=tbl.to_json(orient='records')
            json_data=json.loads(json_str)[0]
            for key, value in json_data.items():
                if '$$$%%%&&&' in value:
                    json_data[key] = json_data[key].strip().split('$$$%%%&&&')

            for key, value in json_data.items():
                if  'Date Revised' not in key:
                    self.json[key.strip().replace(':','')] = json_data[key]
            return True
        else:
            return  False

    def html_to_df_json(self, html_str, tb, stri=''):
        """Get attributes
        '1.Primary Job Information & Mobility',
        '2.Frequently Used Tools's,
        '3.Environmental Conditions',
        '4.Physical Demands',
        '5.Physical Requirements'.
        """
        html = BeautifulSoup(html_str, 'html.parser')
        if tb in [12,13,14,15,16]:
            tbl =  self.tbl_opts_df(html, tb)
            json_str=tbl.to_json(orient='records')
            json_data=json.loads(json_str)[0]
        elif tb in [4,5,6,7,8]:
            tbl = self.tbls_reqd_assets_df(html, tb, stri)
            json_str=tbl.to_json(orient='records')
            json_data=json.loads(json_str)[0]
            if '$$$%%%&&&' in json_data[stri]:
                json_data[stri] = json_data[stri].strip().split('$$$%%%&&&')
            elif len(json_data[stri]) > 0:
                json_data[stri] = [json_data[stri]]
            elif len(json_data[stri]) == 0:
                 json_data[stri] = ['']
        else:
            return {}
        return  json_data

    def husky_data_to_json(self, file):
        """Conversion of Husky data to Json format
        serial conversion
        """
        import uuid
        paras = []
        self.json={}
        status = {}
        if '$$$%%%&&&' in str(file):
           filename=file.split('$$$%%%&&&')[0]
           html=file.split('$$$%%%&&&')[1]
        else:
            filename=file.filename
            html=mammoth.convert_to_html(file).value
            
        self.json['SOURCE'] = 'HUSKY'
        self.json['LineItemID'] = ''
        self.json['DateTime'] = ''
        f = filename.replace('.DOCX', '').replace('.docx', '')
        #print (filename)
        self.json['FILENAME'] = filename
        self.json['ADDED_BY'] = ''
        self.json['GUID'] = str(uuid.uuid4())
        result=self.html_to_df(html)
        if result:
            keys = list(self.json.keys())
            pt = False
            for i in ['Reports To (Position)','Safety-Sensitive','Location (city/site)',
            'Title (and #) of Direct Reports',
            "Date Revised  (Select today's date)",'Position Title', "POSITION SUMMARY"]:
                if i == 'Position Title' and i in keys:
                   self.json['PT'] =  self.json[i]
                   pt = True
                if i == "POSITION SUMMARY" and i in keys:
                   self.json['PS'] =  self.json[i]
                if i in keys:
                    del self.json[i]
            if pt and self.json['PT'] != '':
                ca_json = self.html_to_df_json(html, 4, 'CA')
                cp_json = self.html_to_df_json(html, 5, 'CP')
                er_json = self.html_to_df_json(html, 6, 'ER')
                ei_json = self.html_to_df_json(html, 7, 'EI')
                certifications = ca_json['CA'] + cp_json['CP']
                self.json['CERTIFICATION']  = [x for x in certifications if x != '']
                experience = er_json['ER'] + ei_json['EI']
                self.json['EXPERIENCE']  = [x for x in experience if x != '']
                skills = self.html_to_df_json(html, 8, 'TS')
                self.json['TS'] =skills['TS']
    
                #dictionary=html_to_df(i, 9)
                #json_jp.update(dictionary)
                #dictionary=html_to_df_json(i, 12)
                #paras.append([dictionary])
                #json_jp['Primary Job Information & Mobility'] = paras[-1]
                #dictionary=html_to_df_json(i, 13)
                #paras.append([dictionary])
                #json_jp['Frequently Used Tools'] = paras[-1]
                #dictionary=html_to_df_json(i, 14)
                #paras.append([dictionary])
                #json_jp['Environmental Conditions'] = paras[-1]
                #dictionary=html_to_df_json(i, 15)
                #paras.append([dictionary])
                #json_jp['Physical Demands'] = paras[-1]
                #dictionary=html_to_df_json(i, 16)
                #paras.append([dictionary])
                #json_jp['Physical Requirements'] = paras[-1]
    #            try:
    #                os.mkdir('json_files')
    #            except:
    #                   print('Json directory exits')
    
                self.json['STATUS'] = True
                updatedProfileArray=clf.clf(self.json)
                #return  updatedProfileArray
               # with open('json_files/' + f + '.json', 'w') as outfile:
                #    json.dump(self.json, outfile)
                status = Dump_to_json_store.dump_to_json_store(updatedProfileArray)
                if status:
                   return updatedProfileArray
                else:
                     self.json['STATUS'] = False
                     return self.json
            else:
                self.json['STATUS'] = False
                print('Json not in req format or data fields are empty')
                return self.json
        else:
            self.json['STATUS'] = False
            return self.json


if __name__ == '__main__':
    husky_ob=husky_data_json()
    f='C:\\Users\\amit.ajit.magadum\\Desktop\\Husky_taleo_api\\docs\\Completions Superintendent - Ansell - Drilling & Completions.DOCX'
    v=husky_ob.husky_data_to_json(f)
    print(v)
#updated file