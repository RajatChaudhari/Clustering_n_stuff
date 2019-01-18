import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
#import textblob
#from textblob import Word
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
import nltk
import numpy as np
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import PorterStemmer
st = PorterStemmer()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

df=pd.read_csv('total_profiles.csv')
df = df[['Department: ','Position Title:','POSITION SUMMARY:','responsabilities','EI','CA','TS']]


df=df.fillna(" ")
familypos= df['Department: '] + df['Position Title:'] + df['POSITION SUMMARY:'] + df['responsabilities'] + df['EI'] + df['CA'] + df['TS'] 

print("Total Length:",len(familypos))
familypos=familypos.dropna()
print("Total Length after removing NA:",len(familypos))
print("unique Length:",len(set(familypos)))
#correct = []
#for i in tqdm(set(familypos)): correct.append(''.join(textblob.TextBlob(i).correct()))
#print("unique Length after spelling correction:",len(set(correct)))

word_count = familypos.apply(lambda x: len(str(x).split(" ")))
print("max word count",max(word_count))
print("min word count",min(word_count))
print("max word count",sum(word_count)/len(word_count))
print("average character count",max(familypos.str.len()))
print("min character count",min(familypos.str.len()))
stop_words= familypos.apply(lambda x: len([x for x in x.split() if x in stop]))
print("Total Stop Words",sum(stop_words))
special_character = familypos.apply(lambda x: len([x for x in x.split() if x in ('&','@','#','$','%','!')]))
print("Total special_character",sum(special_character))
numerics = familypos.apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print("Total numbers",sum(numerics))
upper = familypos.apply(lambda x: len([x for x in x.split() if x.isupper()]))
print("Total upprfamilyposse",sum(upper))

####
familypos = familypos.apply(lambda x: " ".join(x.lower() for x in x.split()))
familypos = familypos.str.replace('[^\w\s]','')
familypos = familypos.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
####

cfreq = pd.Series(' '.join(familypos).split()).value_counts()[:10]
print("\n\nCommon words and count:\n",cfreq,'\n\n')
rfreq = pd.Series(' '.join(familypos).split()).value_counts()[-10:]
print("\n\nrare words and count:\n",rfreq,'\n\n')

####
familypos=familypos.apply(lambda x: " ".join(x for x in x.split() if x not in rfreq))
#familypos =familypos.apply(lambda x: " ".join[Word(word).lemmatize() for word in x.split()])
####


tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(familypos)]

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


model.save("d2v_familypos.model")

model=Doc2Vec.load("d2v_familypos.model")

familypos_vec = np.array([model.docvecs[i] for i in range(0, len(model.docvecs))])
print("Embeddings original dimension: ", familypos_vec.shape[1])

ide_familypos = pd.DataFrame(familypos)

ide_familypos['familypos_vec']=familypos_vec.tolist()
ide_familypos['Department: ']=df['Department: ']
mean_vecs = np.mean(familypos_vec, axis=0)
std_vecs = np.std(familypos_vec, axis=0)
familypos_data = (familypos_vec - mean_vecs) / std_vecs

ca_model = PCA(random_state=15, whiten=True, svd_solver='full').fit(data)
n_pca_dim = 250
print(f'Cumulative explained variation for {n_pca_dim} dimensions: {int(np.sum(pca_model.explained_variance_ratio_[:n_pca_dim])*100)} %')
pca_features = pca_model.transform(data)[:,:n_pca_dim]
print("Number of dimensions after PCA: ", pca_features.shape)


Tsne = TSNE(n_componenfamilypos=2, verbose=2, perplexity=30, n_iter=250, random_state=15, learning_rate=2000, early_exaggeration=100)
Tsne_results = familyposne.fit_transform(pca_features)



sse = {}
for k in range(1, 45):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(Tsne_results)
    data = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center



plt.figure(figsize=(20,10))
plt.plot(list(sse.keys()), list(sse.values()), )
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


for n_cluster in range(2, 40,2):
    kmeans = KMeans(n_clusters=n_cluster).fit(Tsne_results)
    label = kmeans.labels_
    sil_coeff = silhouette_score(familypos_vec, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))


ms = KMeans(n_clusters=39, random_state=10) 
ms.fit(Tsne_results)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

joblib.dump(ms,'kmeans_familypos_cluster.pkl')


#ms = joblib.load('kmean_familypos_cluster.pkl')
print(labels.shape)
print("number of estimated family : %d" % n_clusters_)


ide_familypos['label']= labels.tolist()



simil={}
cfamily={}
mfamily={}
for i in range(len(cluster_centers)):
    simil={}
    temp = ide_familypos[ide_familypos['label'] == i]
    for j in list(temp.index):
        val=cosine(np.asarray(temp['familypos_vec'][j]),cluster_centers[i])
        simil[temp['Department: '][j]] = val
    fam=list(simil.keys())[list(simil.values()).index(max(simil.values()))]
    cfamily[i]= fam
    mfamily[fam] = simil

cfamily

df['label']= ide_familypos['label']
df['family']=df['label'].map(cfamily)
joblib.dump(cfamily,'families.lisfamilypos')

df.to_csv('familypos_gen_family_kmeans.csv')
print(df.info())

cfamily

