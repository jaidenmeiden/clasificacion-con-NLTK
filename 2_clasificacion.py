#!/usr/bin/env python
# coding: utf-8

# # Clasificación de palabras (por género de nombre en Español)

# ### Ejercicio de práctica
# 
# **Objetivo:** Construye un classificador de nombres en español usando el siguiente dataset: 
# https://github.com/jvalhondo/spanish-names-surnames

# 1. **Preparación de los datos**: con un `git clone` puedes traer el dataset indicado a tu directorio en Colab, luego asegurate de darle el formato adecuado a los datos y sus features para que tenga la misma estructura del ejemplo anterior con el dataset `names` de nombres en ingles. 
# 
# * **Piensa y analiza**: ¿los features en ingles aplican de la misma manera para los nombres en español?

# In[ ]:


import nltk, random


# In[16]:


get_ipython().system('git clone https://github.com/jvalhondo/spanish-names-surnames')


# In[17]:


def atributos(palabra):
    return {'Ultima letra': palabra[-1]}

def mas_atributos(nombre):
    atributos = {}
    atributos["primera_letra"] = nombre[0].lower()
    atributos["ultima_letra"] = nombre[-1].lower()
    for letra in 'abcdefghijklmnopqrstuvwxyz':
        atributos["count({})".format(letra)] = nombre.lower().count(letra)
        atributos["has({})".format(letra)] = (letra in nombre.lower())
    return atributos

def atributos_esp(nombre):
    atributos = {}
    atributos["primera_letra"] = nombre[0].lower()
    atributos["ultima_letra"] = nombre[-1].lower() #Ultima letra
    #atributos["ultimas_cinco_letras"] = nombre[-5:].lower() #ultimas 5 letras
    atributos["palabras"] = len(nombre.split(" ")) #Cuantas palabras tiene el nombre
    for i, palabra in enumerate(nombre.split(" ")):
        atributos["primera_letra({})".format(i)] = palabra[0].lower()
        atributos["ultima_letra({})".format(i)] = palabra[-1].lower()
    return atributos


# In[18]:


import numpy as np
tag_men = np.genfromtxt('spanish-names-surnames/male_names.csv', skip_header=1, delimiter=',', dtype=('U20','i8','f8'))
tag_women = np.genfromtxt('spanish-names-surnames/female_names.csv', skip_header=1, delimiter=',', dtype=('U20','i8','f8'))


# In[19]:


tagsetE = [(name[0], 'male') for name in tag_men] + [(name[0], 'female') for name in tag_women]
len(tagsetE)


# In[20]:


random.shuffle(tagsetE)
tagsetE


# In[21]:


atributos_esp('DIANA MARISOL')


# In[22]:


atributos_esp('VASILKA')


# 2. **Entrenamiento y performance del modelo**: usando el classificador de Naive Bayes de NLTK entrena un modelo sencillo usando el mismo feature de la última letra del nombre, prueba algunas predicciones y calcula el performance del modelo. 

# In[23]:


# escribe tu código aquí
fsetE1 = [(atributos(n), g) for (n, g) in tagsetE]
trainE1, testE1 = fsetE1[10000:], fsetE1[:10000]
print(len(trainE1))
print(len(testE1))


# In[24]:


classifierE1 = nltk.NaiveBayesClassifier.train(trainE1)
print(nltk.classify.accuracy(classifierE1, trainE1))
print(nltk.classify.accuracy(classifierE1, testE1))


# In[25]:


fsetE2 = [(mas_atributos(n), g) for (n, g) in tagsetE]
trainE2, testE2 = fsetE2[10000:], fsetE2[:10000]


# In[26]:


classifierE2 = nltk.NaiveBayesClassifier.train(trainE2)
print(nltk.classify.accuracy(classifierE2, trainE2))
print(nltk.classify.accuracy(classifierE2, testE2))


# 3. **Mejores atributos:** Define una función como `atributos_esp()` donde puedas extraer mejores atributos con los cuales entrenar una mejor version del clasificador. Haz un segundo entrenamiento y verifica como mejora el performance de tu modelo. ¿Se te ocurren mejores maneras de definir atributos para esta tarea particular?

# In[27]:


fsetE3 = [(atributos_esp(n), g) for (n, g) in tagsetE]
trainE3, testE3 = fsetE3[10000:], fsetE3[:10000]
print(len(trainE3))
print(len(testE3))


# In[28]:


classifierE3 = nltk.NaiveBayesClassifier.train(trainE3)
print(nltk.classify.accuracy(classifierE3, trainE3))
print(nltk.classify.accuracy(classifierE3, testE3))


# In[34]:


get_ipython().system('jupyter nbconvert --to=python 2_clasificacion.ipynb')

