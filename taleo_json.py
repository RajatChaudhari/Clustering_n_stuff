'''
Converting each job profile from husky(docx) data to json.
Processing raw text to clean json format to be used for nlp processing.
'''
from pandas import DataFrame, concat
from os import listdir, walk
from os.path import join as join_path
from glob import glob
import json
import datetime
import Dump_to_json_store
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from preprocess_json import tokenize_text
from pandas import DataFrame, ExcelFile, read_csv, read_excel
import json,os
import family_classifier
import pandas as pd
import uuid

clf = family_classifier.family_classify()

class taleo_to_json(object):
    """Processing Taleo data into json format"""
    def __init__(self, filename):
       self.filename = filename
       
    def tokenize_text(self, text):
       """Tokenise text"""
       return word_tokenize(text)

    def transform_data_to_tfidf(self, texts, tfidf_model):
        strs = []
        for text in texts:
            stra = self.tokenize_text(text)
            stra = ' '.join(stra)
            strs.append(stra)
        response = tfidf_model.transform(strs)
        return response

    def taleo_data_to_json(self, file_read):
        """Convert Taleo data to json format"""
        tfidf_model = joblib.load('model/tfidf_model.pkl')
        section_identifier = joblib.load('model/classification_text.pkl')
        jsons = []
        guid = str(uuid.uuid4())
        if isinstance(file_read, pd.DataFrame):
            taleo_db = file_read
        else:
            taleo_db = read_excel(file_read)
        output_df = DataFrame()
        columns = list(taleo_db.columns)
        if 'Requisition Title' in columns and 'Req. Identifier' in columns:
            Req_identifier = taleo_db['Req. Identifier']
            output_df['PT'] = taleo_db['Requisition Title']
            output_df['Department'] = taleo_db['Job Family']

            taleo_db['External: Responsibilities'] = taleo_db['External: Responsibilities'].fillna('')

            output_df['responsibilities'] = taleo_db['External: Responsibilities'].apply(lambda x: '$$$%%%&&&'.join(x.strip().split('-')[1:]) if x else None)
            output_df['PS'] = taleo_db['Original Description Section - External']
            output_df['SOURCE'] = 'TALEO'
            output_df['GUID'] = guid
            output_df['FILENAME'] = self.filename
            output_df['ADDED_BY'] = ''
            taleo_db['Qualifications - External'] = taleo_db['Qualifications - External'].fillna('')

            taleo_db1 = output_df[['GUID', 'FILENAME', 'SOURCE','ADDED_BY', 'PT','Department','responsibilities','PS']]
            json_taleo = taleo_db1.to_json(orient='records')
            json_taleo1 = json.loads(json_taleo)
            lis=['CA','CP','TS','EI','ER']

            def categorize_items(line):
                """Categorising Taleo attributes"""
                items = line.split(' - ')[1:]
                dit={}
                if items:
                    items_tfidf = self.transform_data_to_tfidf(items, tfidf_model)
                    labels = [section_identifier.predict(items_tfidf) for item in items]
                    for i in lis:
                        dit[i] = '$$$%%%&&&'.join([item.strip() for item, label in zip(items, labels[0]) if label == i])
                return dit

            for i, j in enumerate(json_taleo1):
                zx=categorize_items(taleo_db['Qualifications - External'][i])
                keys = list(zx.keys())
                for key in lis:
                    if key in keys:
                       if len(zx[key]) == 0:
                          json_taleo1[i][key] = ['']
                       else:
                            json_taleo1[i][key] = zx[key].strip().split('$$$%%%&&&')
                    else:
                        json_taleo1[i][key] = ['']
                certifications = json_taleo1[i]['CA'] + json_taleo1[i]['CP']
                experience = json_taleo1[i]['EI'] + json_taleo1[i]['ER']
                json_taleo1[i]['CERTIFICATION'] = [x for x in certifications if x != '']
                json_taleo1[i]['EXPERIENCE'] = [x for x in experience if x != '']
                del json_taleo1[i]['CA']
                del json_taleo1[i]['CP']
                del json_taleo1[i]['EI']
                del json_taleo1[i]['ER']
                if 'responsibilities' in list(json_taleo1[i].keys()) and type(json_taleo1[i]['responsibilities']) != list and \
                (json_taleo1[i]['responsibilities'] is not None):
                    resp_list = json_taleo1[i]['responsibilities'].strip().split('$$$%%%&&&')
                    resp_list = [st.strip() for st in resp_list]
                    json_taleo1[i]['responsibilities'] = resp_list
                else:
                    json_taleo1[i]['responsibilities'] = ''
            try:
                os.mkdir('json_files')
            except:
                print('Json directory exits')

            for index, json_data in enumerate(json_taleo1):
                filename = Req_identifier[index]
                lineid = filename
##                if '-' in filename:
##                    filename = filename.replace('-','_')
##                    filename = 'Taleo_' + filename
##                else:
##                    filename = 'Taleo_' + filename
                
                json_data['STATUS'] =  True
                json_data['LineItemID'] = lineid
                json_data['DateTime'] = ''
                
                updatedProfileArray=clf.clf(json_data)
                
                status = Dump_to_json_store.dump_to_json_store(updatedProfileArray)
                if status:
                   jsons.append(updatedProfileArray)
                else:
                    json_data['STATUS'] =  False 
                    jsons.append(json_data)
            return jsons
        else:
            di = {'GUID': guid, 'FILENAME':self.filename, 'STATUS': False}
            return di
if __name__ == '__main__':
    husky_ob=taleo_to_json()
    f='C:\\New folder\\Book1.xlsx'
    v=husky_ob.taleo_data_to_json(f)
    print(v[0])
    print(v[0]['JSON'].keys())

#updated file