import pandas as pd
import textblob
from tqdm import tqdm
df=pd.read_csv('total_profiles.csv')
df = df[['Department: ','Position Title:','POSITION SUMMARY:','responsabilities','EI','CA','CP','TS']]

#df.to_csv('total_profiles_selected.csv')
#df.fillna("", inplace=True)

#df=df.dropna()

#df['Department: '].apply(lambda txt: ''.join(textblob.TextBlob(txt).correct()))
#[df[i].apply(lambda txt: ''.join(textblob.TextBlob(txt).correct())) for i in s]
df.info()

import numpy as np
from sklearn.manifold import TSNE

from nltk.corpus import stopwords
stop = stopwords.words('english')

from nltk.stem import PorterStemmer
st = PorterStemmer()

from textblob import Word
import nltk
import numpy as np
nltk.download('wordnet')
nltk.download('punkt')

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from sklearn.externals import joblib

psummary=df['POSITION SUMMARY:']
print("Total Length:",len(psummary))
psummary=psummary.dropna()
print("Total Length after removing NA:",len(psummary))
print("unique Length:",len(set(psummary)))
#correct = []
#for i in tqdm(set(psummary)): correct.append(''.join(textblob.TextBlob(i).correct()))
print("unique Length after spelling correction:",len(set(correct)))
word_count = psummary.apply(lambda x: len(str(x).split(" ")))
print("max word count",max(word_count))
print("min word count",min(word_count))
print("max word count",sum(word_count)/len(word_count))
print("average character count",max(psummary.str.len()))
print("min character count",min(psummary.str.len()))
stop_words= psummary.apply(lambda x: len([x for x in x.split() if x in stop]))
print("Total Stop Words",sum(stop_words))
special_character = psummary.apply(lambda x: len([x for x in x.split() if x in ('&','@','#','$','%','!')]))
print("Total special_character",sum(special_character))
numerics = psummary.apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print("Total numbers",sum(numerics))
upper = psummary.apply(lambda x: len([x for x in x.split() if x.isupper()]))
print("Total upprcase",sum(upper))

####
psummary = psummary.apply(lambda x: " ".join(x.lower() for x in x.split()))
psummary = psummary.str.replace('[^\w\s]','')
psummary = psummary.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
####

cfreq = pd.Series(' '.join(psummary).split()).value_counts()[:10]
print("\n\nCommon words and count:\n",cfreq,'\n\n')
rfreq = pd.Series(' '.join(psummary).split()).value_counts()[-10:]
print("\n\nrare words and count:\n",rfreq,'\n\n')

####
psummary=psummary.apply(lambda x: " ".join(x for x in x.split() if x not in rfreq))
psummary = psummary.apply(lambda x: " ".join[Word(word).lemmatize() for word in x.split()]))
####


tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(psummary)]

max_epochs = 100
vec_size = max(word_count)
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    model.alpha -= 0.0002


model.save("d2v_psummary.model")

model=Doc2Vec.load("d2v_psummary.model")

psummary_vec = np.array([model.docvecs[i] for i in range(0, len(model.docvecs))])
print("Embeddings original dimension: ", psummary_vec.shape[1])

ide_psummary = pd.DataFrame(psummary)

ide_psummary['psummary_vec']=psummary_vec.tolist()
ide_psummary['Department: ']=df['Department: ']
#mean_vecs = np.mean(psummary_vec, axis=0)
#std_vecs = np.std(psummary_vec, axis=0)
#psummary_data = (psummary_vec - mean_vecs) / std_vecs

#tsne = TSNE(n_components=2, verbose=2, perplexity=30, n_iter=1000, random_state=15, learning_rate=2000, early_exaggeration=100)
#tsne_results = tsne.fit_transform(psummary_data)

from sklearn.cluster import MeanShift

ms = MeanShift()
ms.fit(psummary_vec)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

joblib.dump(ms,'kmean_psummary_cluster.pkl')

ms = joblib.load('kmean_psummary_cluster.pkl')
print(labels.shape)
print("number of estimated family : %d" % n_clusters_)


ide_psummary['label']= labels.tolist()

#print(labels.tolist().index(cluster_centers))


from scipy.spatial.distance import cosine
simil={}
cfamily={}
mfamily={}
for i in range(len(cluster_centers)):
    simil={}
    temp = ide_psummary[ide_psummary['label'] == i]
    for j in list(temp.index):
        val=cosine(np.asarray(temp['psummary_vec'][j]),cluster_centers[i])
        simil[temp['Department: '][j]] = val
    fam=list(simil.keys())[list(simil.values()).index(max(simil.values()))]
    cfamily[i]= fam
    mfamily[fam] = simil

cfamily

df['label']= ide_psummary['label']
df['family']=df['label'].map(cfamily)
joblib.dump(cfamily,'families.lists')
df.to_csv('psummary_gen_family.csv')
print(df.info())







