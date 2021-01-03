#!/usr/bin/env python
# coding: utf-8

# # Clasificación de palabras (por género de nombre)

# In[1]:


import nltk, random
nltk.download('names')
from nltk.corpus import names 


# **Función básica de extracción de atributos**
# 
# Creamos una función donde nos devuelve unicamente la última letra de cada palabra

# In[2]:


# definición de atributos relevantes
def atributos(palabra):
    return {'Ultima letra': palabra[-1]}


# Unimos la lista de nombres **masculinos** y **femeninos** en una lista de tuplas

# In[3]:


tagset = [(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')]
len(tagset)


# Hacemos que la lista sea distribuida aleatoriamente para evitar el sesgo, ya que la lista de nombres masculinos queda primero que la lista de nombres femeninos.

# In[4]:


random.shuffle(tagset)
tagset[:10]


# Creo una segunda lista de atributos, ya que nuestro modelo NO lee los nombres, sino los atributos de los nombres.

# In[5]:


fset1 = [(atributos(n), g) for (n, g) in tagset]


# Divido el dataset en los conjuntos de entrenamiento y pruebas

# In[6]:


train1, test1 = fset1[2000:], fset1[:2000]
print(len(train1))
print(len(test1))


# **Modelo de clasificación Naive Bayes**

# In[7]:


# entrenamiento del modelo NaiveBayes
classifier = nltk.NaiveBayesClassifier.train(train1)


#  **Verificación de algunas predicciones**

# In[8]:


classifier.classify(atributos('amanda'))


# In[9]:


classifier.classify(atributos('peter'))


# **Performance del modelo**

# In[10]:


print(nltk.classify.accuracy(classifier, train1))


# In[11]:


print(nltk.classify.accuracy(classifier, test1))


# **Mejores atributos**
# 
# Los atributos se almacenan en un diccionario. Creo una función de atributos completamente personalizada (Hay que recordar que no hay un protocolo establecido para definir atributos).
# 
# Para este caso:
# * Atributo 1: La primera letra de la palabra
# * Atributo 2: La última letra de la palabra
# * Atributo 3: Cuantas veces aparece una letra del alfabeto en la palabra
# * Atributo 4: Si el alfabeto tiene o no tiene una letra del alfabeto

# In[12]:


def mas_atributos(nombre):
    atributos = {}
    atributos["primera_letra"] = nombre[0].lower()
    atributos["ultima_letra"] = nombre[-1].lower()
    for letra in 'abcdefghijklmnopqrstuvwxyz':
        atributos["count({})".format(letra)] = nombre.lower().count(letra)
        atributos["has({})".format(letra)] = (letra in nombre.lower())
    return atributos


# In[13]:


mas_atributos('jhon')


# In[14]:


fset2 = [(mas_atributos(n), g) for (n, g) in tagset]
train2, test2 = fset2[2000:], fset2[:2000]


# In[15]:


classifier2 = nltk.NaiveBayesClassifier.train(train2)


# In[16]:


print(nltk.classify.accuracy(classifier2, train2))


# In[17]:


print(nltk.classify.accuracy(classifier2, test2))

