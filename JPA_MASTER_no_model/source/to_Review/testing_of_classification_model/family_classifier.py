from sklearn.externals import joblib
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import mammoth



class family_classify:
    def __init__(self):
        self.classifier = joblib.load('model/sgd_family_classifier.model')
        self.family_dict = joblib.load('model/family_dict.pkllist')

        
    def clf(self,family):
        test= family['Department: '] + family['Position Title:'] + family['POSITION SUMMARY:'] + family['responsabilities'] + family['EI'] + family['CA'] + family['CP'] + family['TS'] 
        return (self.family_dict[self.classifier.predict(test)[0]])



