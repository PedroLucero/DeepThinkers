#!/usr/bin/env python
# coding: utf-8

# In[1]:


# setiando el uso de la grafica para el modelo
#gpus = tf.config.experimental.list_physical_devices('GPU')

# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


# In[2]:


#Librerias
#pip install nltk
#pip install pickle5
#pip install tensorflow


# ### Importacionde librerias

# In[3]:


import json

import pickle
import numpy as np
import re
import nltk

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense,Dropout
from tensorflow.keras.optimizers import SGD


# ### Diccionario de conocimiento:

# In[4]:


def guardar_json(datos, filename):
    '''creacion de funcion para guardar 
    diccionario de conocimiento en formato json'''
    archivo=open(filename,"w")
    json.dump(datos,archivo,indent=4)
    pattern = r"\b[a-z|0-9]{6,64}@[a-z|0-9]{1,254}.[a-z|0-9]{0,10}\b"

my_str = "my email is adga2002@hotmail.com"

def check_email(pattern, my_str):
    match = re.search(pattern, my_str)

    if match:
        print("Gracias por su correo!")
        print(match.group())
        return ["Guardar el dato"]

    print("No te entendí bien...")
    return []
    

biblioteca = {"intents": 
                [{"tag":"saludos",
                 "patterns":['hola','buenos dias','buenas tardes', 'Buenas noches','buenas','Buenas','buen dia'],
                 "responses":['¡Bienvenido al asistente virtual de DeepThinkers! \n¿Se encuentra afiliado a nuestro sistema?'],
                 "context":[""]
                 }, 
                 
                    {"tag":"afiliadon",
                 "patterns":['no estoy afiliado','no','negativo'],
                 "responses":['Indíquenos su correo electrónico para su registro.'],
                 "context":[""]
                 }, 
                 
                 {"tag":"afiliadop",
                 "patterns":['si estoy afiliado','si','correcto'],
                 "responses":['Indíquenos su correo electrónico, por favor.'],
                 "context":[""]
                 }, 
                 
                 {"tag":"usuarios",
                 "patterns":['my email is adga2002@hotmail.com'],
                 "responses":['¡Verificado!. ¿Desea comprar o ver productos?'],
                 "context":[""]
                 }, 
                 
                 {"tag":"comprar",
                 "patterns":['comprar', 'quiero comprar'],
                 "responses":['Inserte el producto que desea comprar.'],
                 "context":[""]
                 }, 
                 
                 {"tag":"productos",
                 "patterns":['guineo',
                             'manzana',
                              'pan',
                             'queso',
                             'pollo',
                             'cafe',
                              'leche'],
                 "responses":['Indique la cantidad que desea comprar.'],
                 "context":[""]
                 }, 
                 
                 {"tag":"cantidad",
                 "patterns":['1','2','3','4','5','6','7','8','9','10','libra','libras','gramos','kilos','litros','litro' ],
                 "responses":['¡Gracias por su compra!'],
                 "context":[""]
                 }, 
                 
                 {"tag":"despedidas",
                 "patterns":['chao','adios','hasta luego','nos vemos','hasta la proxima'],
                "responses":['¡Un placer atenderte del equipo de DeepThinkers! Estamos atentos a cualquier duda adicional.'],
                "context":[],
                 },
                
                 {"tag":"agradecimientos",
                 "patterns":["gracias",
                             "muchas gracias",
                             "mil gracias",
                             "muy amable",
                             "se lo agradezco",
                             "fue de ayuda",
                             "gracias por la ayuda",
                             "muy agradecido",
                             "gracias por su tiempo"                             
                            ],
                 "responses":["De nada.",
                              "¡Feliz por ayudarle!",
                              "Gracias a usted.",
                              "Estamos para servirle!",
                              "Fue un placer :)"
                             ],
                 "context":[""]
                },
                 
                {"tag":"norespuesta",
                 "patterns":[" "],
                 "responses":["No se ha detectado una respuesta válida."
                             ],
                 "context":[""]                    
                }
                ]            
                }
# Guardado de diccionario de conocimiento en formato json.
guardar_json(biblioteca, 'intents.json')
biblioteca


# ### Construccion de bolsa de palabras, clases y documents

# In[5]:


bolsadepalabras = []
clases = [] 
documents = []
for intent in biblioteca['intents']:
    
    clases.append(intent['tag'])
    
    for pattern in intent['patterns']:       
        result = nltk.word_tokenize(pattern)        
        bolsadepalabras.extend(result)
        
        documents.append((result, intent['tag']))
    
print(bolsadepalabras)     
print(clases)

for elemento in documents:
    print('\n')
    print(elemento)


# ### Aplicación stemmer y limpieza sobre la bolsa de palabras

# In[6]:


from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

ignore_words = ["?","¿","!","¡","."]                  # Lista de simbolos que se desean eliminar.

bolsadepalabras2 = []                                 # definicion de variable auxiliar para guardar el resultado limpio

for w in bolsadepalabras:                             # iteracion sobre la lista de palabras "bolsadepalabras"
    if w not in ignore_words:
        wprocesada = w.lower()                        # convertir a minuscula       
        wprocesada = stemmer.stem(wprocesada)         # para stemmer 
        bolsadepalabras2.append(wprocesada)           # agregar a la lista.
        
print("bolsadepalabras2:", bolsadepalabras2)


# #### Compresión de listas

# In[7]:


bolsadepalabras = [stemmer.stem(w.lower()) for w in bolsadepalabras if w not in ignore_words]
print(len(bolsadepalabras))
print(bolsadepalabras)


# ### Uso de la funcion split y join para eliminar espacios dobles

# In[8]:


sentencia = 'HOLAA    como estas?'         # string de entrada 
print(sentencia.split())                   # aplicacion del metodo split para obtener una lista de los elementos por separado.
sentencia= ' '.join(sentencia.split())     # aplicacion del metodo join para unir la lista anterior
print(sentencia)
sentencia = sentencia.lower()              # conversion a minusculas
print(sentencia)


# ### Creación de set de entramiento (Vector de entrada y vector de salida) 

# In[9]:


def cleanString(words, ignore_words):
    '''funcion utilizada para limpiar lista de palabras,
     el uso de funciones, evita repetir la innecesaria de codigo'''
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    return words


bolsadepalabras = cleanString(bolsadepalabras, ignore_words)
pickle.dump(bolsadepalabras,open("bolsadepalabras.pkl","wb")) # guarda bolsa de palabras como archivo .pkl
pickle.dump(clases,open("classes.pkl","wb"))        # guarda lista de clases como archivo .pkl
training = [] # Creacion de lista vacia de para agregar los vectores construidos en las siguientes lineas.

for doc in documents:    
    
    interaccion = doc[0]            # obtencion del primer elemento guardado en cada posicion de la lista documents.
    interaccion = cleanString(interaccion, ignore_words) # limpieza del strin "interaccion"
    
    entradacodificada = []  # creacion de la lista vacia llamada "entradacodificada"
    
    # codificacion de la entrada
    for palabra in bolsadepalabras:
        if palabra in interaccion:
            entradacodificada.append(1)
        else:
            entradacodificada.append(0)    
    
    # codificacion de la etiqueta
    salidacodificada = [0]*len(clases)
    indice = clases.index(doc[1])
    salidacodificada[indice] = 1
    
    training.append([entradacodificada, salidacodificada])
    
training = np.array(training, dtype=list)

x_train = list(training[:,0])

y_train = list(training[:,1])

print(x_train[0])
print(len(x_train[0]))


# ### Creacion de nuestra red neuronal

# In[10]:


model = Sequential()

model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu')) #capa oculta -> aprendizaje
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))


# ### Entrenamiento de nuestra red neuronal

# In[11]:


sgd = SGD(learning_rate=0.01,momentum=0.9,nesterov=True) # ,decay=1e-6

model.compile(loss="categorical_crossentropy", optimizer=sgd,metrics=["accuracy"])

hist = model.fit(np.array(x_train),np.array(y_train),epochs=300,batch_size=5,verbose=True)
model.save("chatbot_model.h5",hist)
print("modelo creado")


# ### Carga de modelo y archivos necesarios

# In[12]:


from tensorflow.keras.models import load_model
import json
import pickle
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

ignore_words = ["?","¿","!","¡","."]
model = load_model("chatbot_model.h5")
biblioteca = json.loads(open("intents.json").read())
bolsadepalabras = pickle.load(open("bolsadepalabras.pkl","rb"))
clases = pickle.load(open("classes.pkl","rb"))

print(len(bolsadepalabras))


# ### Creación de funciones: limpieza de entrada y binarización de la entrada
# 

# In[13]:


def cleanEntrada(entradaUsuario):
    entradaUsuario = nltk.word_tokenize(entradaUsuario)
    entradaUsuario = [stemmer.stem(w.lower()) for w in entradaUsuario if w not in ignore_words]
    return entradaUsuario

def convVector(entradaUsuario, bolsadepalabras):
    
    entradaUsuario = cleanEntrada(entradaUsuario)
    
    vectorentrada = [0]*len(bolsadepalabras)    # colocar vector de entrada como ceros    
    for palabra in entradaUsuario:              # loop sobre la entrada del usuario
        
        if palabra in bolsadepalabras:          # verificación si la palabra esta dentro de la bolsa de palabras.
            
            indice = bolsadepalabras.index(palabra)    # obtanción del indice de la palabra actual, en la bolsa de palabras
            vectorentrada[indice] = 1                  #  asignación de 1 en el vector de entrada para el indice correspondiente.
            
    vectorentrada = np.array(vectorentrada)            #  conversión a un arreglo numpy
    return vectorentrada

entradausuario ="buenos dias"
vectorentrada = convVector(entradausuario, bolsadepalabras)
vectorentrada


# ###  Prueba de nuestra red neuronal sobre la entra del usuario binarizada 

# In[14]:


def gettag(vectorentrada, LIMITE = 0):
    vectorsalida = model.predict(np.array([vectorentrada]))[0]

    # cargar los indices y los valores retornados por el modelo
    vectorsalida = [[i,r] for i,r in enumerate(vectorsalida) if r > LIMITE]    

    # ordenar salida en funcion de la probabilidad, valor que está contenido en el segundo termino de cada uno de sus elementos.
    vectorsalida.sort(key=lambda x: x[1], reverse=True) 
    print(vectorsalida)
    
    listEtiquetas = []    
    for r in vectorsalida:   
        listEtiquetas.append({"intent": clases[r[0]], "probability": str(r[1])})   
    return listEtiquetas

listEtiquetas = gettag(vectorentrada, LIMITE = 0.1)
listEtiquetas


# ### Función para retornar respuesta

# In[15]:


import random

def getResponse(listEtiquetas, biblioteca):
    etiqueta = listEtiquetas[0]['intent']

    listadediccionarios = biblioteca['intents']

    for dicionario in listadediccionarios:

        if etiqueta == dicionario['tag']:
            listaDeRespuestas = dicionario['responses']            
            respuesta = random.choice(listaDeRespuestas)
            break
    return respuesta

respuesta = getResponse(listEtiquetas, biblioteca)
respuesta


# ### ChatbotRespuesta

# In[16]:


def chatbotRespuesta(entradaUsuario):
    vectorentrada = convVector(entradaUsuario, bolsadepalabras)
    listEtiquetas = gettag(vectorentrada, LIMITE = 0)
    respuesta = getResponse(listEtiquetas, biblioteca)
    return respuesta


# ### Interacción con el usuario

# In[ ]:


entradaUsuario = ''
if __name__ == "__main__":
    while entradaUsuario!='exit':
        entradaUsuario = input()
        respuesta = chatbotRespuesta(entradaUsuario)
        print(respuesta)  


# In[ ]:





# In[ ]:





# In[ ]:




