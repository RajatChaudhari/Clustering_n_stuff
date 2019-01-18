from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from pandas import DataFrame


def tokenize_text(text):
        return word_tokenize(text)


class jp_to_struct:
    def __init__(self):
        self.tfidf_model = joblib.load('tfidf_model.pkl')
        self.section_identifier = joblib.load('Taleo_profile_extractor.pkl')
        self.working_conditions =  ['CA_ASSET','CA_REQD','CP_ASSET','CP_REQD','EI_ASSET','EI_REQD','ER_ASSET','ER_REQD','TS_ASSET','TS_REQD',
                                    'Location (city/site):', 'Reports To (Position):', 'Safety-Sensitive:','Title (and #) of Direct Reports:',
                                    'competency_group','Bending/Crouching:', 'Climbing:', 'Driving:', 'Keyboarding:','Kneeling/Crawling:',
                                    'On-call:', 'Operating equipment:','Sedentary/Sitting:', 'Shift work:', 'Travel:', 'Walking:','Manual tools:',
                                    'Office equipment:','Pneumatic tools:','Power tools:','Shop tools:','Tool belt worn:','Vibration tools:',
                                    'Welding:', ' Extreme heat/cold:', 'Chemicals:', 'Confined spaces:','Heights:','Moving equipment:',
                                    'Night time:', 'Noise:', 'Outdoors:','Rotating equipment:','Toxic gases:','Uneven surfaces:','Wet or damp:',
                                    'Light 11-20 pounds:', 'Medium 21-50 pounds:','Sedentary 0-10 pounds:','Carrying:','Lifting:','Pushing/pulling:',
                                    'Standing:']

    def tokenize_text(self,text):
        return word_tokenize(text)

    def transform_data_to_tfidf(self,texts):
        strs = []
        for text in texts:
            stra = self.tokenize_text(text)
            stra = ' '.join(stra)
            strs.append(stra)
            response = self.tfidf_model.transform(strs)
        return response

    def categorize_items(self,line, section):
        items = line.split(' - ')[1:]
        if items:
            items_tfidf = self.transform_data_to_tfidf(items)
            labels = [self.section_identifier.predict(items_tfidf) for item in items]
            return ('$$$%%%&&&'.join([item for item, label in zip(items, labels[0]) if label == section]))
        return None

    def to_struct(self,db):
        output_df = DataFrame()
        output_df['Position Title'] = getattr(db, 'Requisition Title')
        output_df['Department'] = db['Job Family']
        db['External: Responsibilities'] = db['External: Responsibilities'].fillna('')
        output_df['responsabilities'] = db['External: Responsibilities'].apply(lambda x: '$$$%%%&&&'.join(x.strip().split('-')[1:]) if x else None)
        output_df['POSITION SUMMARY'] = db['Original Description Section - External'] 
        output_df['Date Revised'] = db['Req. Creation Date']
        db['Qualifications - External'] = db['Qualifications - External'].fillna('')
        output_df['CA'] =  db['Qualifications - External'].apply(lambda x: self.categorize_items(x, 'CA'))
        output_df['CP'] =  db['Qualifications - External'].apply(lambda x: self.categorize_items(x, 'CP'))
        output_df['EI'] =  db['Qualifications - External'].apply(lambda x: self.categorize_items(x, 'EI'))
        output_df['ER'] =  db['Qualifications - External'].apply(lambda x: self.categorize_items(x, 'ER'))
        output_df['TS'] =  db['Qualifications - External'].apply(lambda x: self.categorize_items(x, 'TS'))
        for working_condition in self.working_conditions: output_df[working_condition] = ""
        return output_df


