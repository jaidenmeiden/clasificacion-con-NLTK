#!/usr/bin/env python
# coding: utf-8

# # Clasificación de documentos (email spam o no spam)

# In[1]:


get_ipython().system('git clone https://github.com/pachocamacho1990/datasets')


# In[2]:


import nltk, random
import pandas as pd
import numpy as np
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize


# In[3]:


df = pd.read_csv('datasets/email/csv/spam-apache.csv', names = ['clase','contenido'])
df['tokens'] = df['contenido'].apply(lambda x: word_tokenize(x))
print(len(df))
df.head()


# In[4]:


df['tokens'].values[0]


# In[5]:


all_words = nltk.FreqDist([w for tokenlist in df['tokens'].values for w in tokenlist])
top_words = all_words.most_common(200)
top_words


# In[6]:


def document_features(document, top_words=top_words):
    document_words = set(document)
    features = {}
    for word, freq in top_words:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# In[7]:


document_features(df['tokens'].values[0])


# Lo primero que hacemos es un conjunto de atributos como una lista de **tuplas**, obteniendo **textos** y **clases**. De esta forma estamos recorriendo dos listas de forma simultanea.
# 
# La función `zip` se utiliza porque se estan recorriendo dos listas de forma simultanea.

# In[8]:


fset = [(document_features(texto), clase) for texto, clase in zip(df['tokens'].values, df['clase'].values)]
random.shuffle(fset)
train, test = fset[:200], fset[200:]


# In[9]:


classifier = nltk.NaiveBayesClassifier.train(train)


# In[10]:


print(nltk.classify.accuracy(classifier, train))


# In[11]:


print(nltk.classify.accuracy(classifier, test))


# In[12]:


df[df['clase']==-1]['contenido']


# In[13]:


classifier.show_most_informative_features(5)


# In[ ]:


get_ipython().system('jupyter nbconvert --to=python 3_clasificacion.ipynb')

