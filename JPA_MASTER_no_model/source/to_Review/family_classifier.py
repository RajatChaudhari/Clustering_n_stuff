import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


df = pd.read_csv('famil_family.csv')

df.fillna(' ')
df.info()


family=df[df['family'] != 'project plannersschedulers' ] #wrong mapped

family_list= family['family'].unique().tolist()
family_dict={}

for i in range(len(family_list)): family_dict[family_list[i]] = i
joblib.dump(family_dict,'family_dict.pkllist')    
family['label'] = df['family'].map(family_dict)


familycf=joblib.load('family_dict.pkllist')
print(familycf)


print(family.info())
family['train'] = family['Department: '] + family['Position Title:'] + family['POSITION SUMMARY:'] + family['responsabilities'] + family['EI'] + family['CA'] + family['CP'] + family['TS'] 


family=family.dropna()


family['train']

family['train'] = family['train'].apply(lambda x: " ".join(x.lower() for x in x.split()))
family['train'] = family['train'].str.replace('[^\w\s]','')

stop = stopwords.words('english')

family['train'] = family['train'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

rfreq = pd.Series(' '.join(family['train']).split()).value_counts()[-10:]
print("\n\nrare words and count:\n",rfreq,'\n\n')


family['train']=family['train'].apply(lambda x: " ".join(x for x in x.split() if x not in rfreq))

train=family.sample(frac=0.8,random_state=200)
test=family.drop(train.index)

train_X = train['train']
train_Y = train['label']
test_X =  test ['train']
test_Y =  test['label']


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_X)

joblib.dump(vectorizer,'family_clf_tfidfmodel.mod')
test_vectors = vectorizer.transform(test_X)

print(train_vectors.shape, test_vectors.shape)


nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(train_X, train_Y)

joblib.dump(nb,'nb_family_classifier.model')

from sklearn.metrics import classification_report
y_pred = nb.predict(test_X)

print('accuracy %s' % accuracy_score(y_pred, test_Y))
print(classification_report(test_Y, y_pred))



sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(train_X, train_Y)

joblib.dump(sgd,'sgd_family_classifier.model')

y_pred = sgd.predict(test_X)


print('accuracy %s' %accuracy_score(y_pred, test_Y))
print(classification_report(test_Y, y_pred))


logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(train_X, train_Y)

joblib.dump(logreg,'logreg_family_classifier.model')

y_pred = logreg.predict(test_X)


print('accuracy %s' % accuracy_score(y_pred, test_Y))
print(classification_report(test_Y, y_pred))

