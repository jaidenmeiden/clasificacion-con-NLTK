#!/usr/bin/env python
# coding: utf-8

# # Clasificación de documentos (email spam o no spam)

# In[1]:


import nltk, random
import pandas as pd
import numpy as np
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
nltk.download('stopwords')


# ## Ejercicio de práctica
# 

# ¿Como podrías construir un mejor clasificador de documentos?
# 
# 0. **Dataset más grande:** El conjunto de datos que usamos fue muy pequeño, considera usar los archivos corpus que estan ubicados en la ruta: `datasets/email/plaintext/` 
# 
# 1. **Limpieza:** como te diste cuenta no hicimos ningun tipo de limpieza de texto en los correos electrónicos. Considera usar expresiones regulares, filtros por categorias gramaticales, etc ... . 
# 
# ---
# 
# Con base en eso construye un dataset más grande y con un tokenizado más pulido. 

# In[2]:


# Descomprimir ZIP
import zipfile
fantasy_zip = zipfile.ZipFile('datasets/email/plaintext/corpus1.zip')
fantasy_zip.extractall('datasets/email/plaintext')
fantasy_zip.close()


# In[3]:


# Creamos un listado de los archivos dentro del Corpus1 ham/spam
from os import listdir

path_ham = "datasets/email/plaintext/corpus1/ham/"
filepaths_ham = [path_ham+f for f in listdir(path_ham) if f.endswith('.txt')]

path_spam = "datasets/email/plaintext/corpus1/spam/"
filepaths_spam = [path_spam+f for f in listdir(path_spam) if f.endswith('.txt')]


# In[4]:


# Creamos la funcion para tokenizar y leer los archivos 

def abrir(texto):
    with open(texto, 'r', errors='ignore') as f2:
        data = f2.read()
        data = word_tokenize(data)
    return data


# In[5]:


# Creamos la lista tokenizada del ham
list_ham = list(map(abrir, filepaths_ham))
# Creamos la lista tokenizada del spam
list_spam = list(map(abrir, filepaths_spam))


# 2. **Validación del modelo anterior:**  
# ---
# 
# una vez tengas el nuevo conjunto de datos más pulido y de mayor tamaño, considera el mismo entrenamiento con el mismo tipo de atributos del ejemplo anterior, ¿mejora el accuracy del modelo resultante?

# 3. **Construye mejores atributos**: A veces no solo se trata de las palabras más frecuentes sino de el contexto, y capturar contexto no es posible solo viendo los tokens de forma individual, ¿que tal si consideramos bi-gramas, tri-gramas ...?, ¿las secuencias de palabras podrián funcionar como mejores atributos para el modelo?. Para ver si es así,  podemos extraer n-gramas de nuestro corpus y obtener sus frecuencias de aparición con `FreqDist()`, desarrolla tu propia manera de hacerlo y entrena un modelo con esos nuevos atributos, no olvides compartir tus resultados en la sección de comentarios. 

# In[6]:


# Separamos las palabras mas comunes
all_words = nltk.FreqDist([w for tokenlist in list_ham+list_spam for w in tokenlist])
top_words = all_words.most_common(250)
top_words


# In[7]:


# Agregamos Bigramas
bigram_text = nltk.Text([w for token in list_ham+list_spam for w in token])
bigrams = list(nltk.bigrams(bigram_text))
top_bigrams = (nltk.FreqDist(bigrams)).most_common(250)
top_bigrams


# In[8]:


def document_featuresEmail(document, top_words=top_words, top_bigrams=top_bigrams):
    document_words = set(document)
    bigram = set(list(nltk.bigrams(nltk.Text([token for token in document]))))
    features = {}
    for word, j in top_words:
        features['contains({})'.format(word)] = (word in document_words)

    for bigrams, i in top_bigrams:
        features['contains_bigram({})'.format(bigrams)] = (bigrams in bigram)
  
    return features


# In[9]:


# Juntamos las listas indicando si tienen palabras de las mas comunes
import random
fset_ham = [(document_featuresEmail(texto), 0) for texto in list_ham]
fset_spam = [(document_featuresEmail(texto), 1) for texto in list_spam]
fset = fset_spam + fset_ham
random.shuffle(fset)
len(fset)


# In[10]:


fset_train, fset_test = fset[:2000], fset[2000:]
# Entrenamos el programa
classifier = nltk.NaiveBayesClassifier.train(fset_train)


# In[11]:


classifier.classify(document_featuresEmail(list_ham[34]))


# In[12]:


print(nltk.classify.accuracy(classifier, fset_test))


# In[13]:


classifier.show_most_informative_features(5)

