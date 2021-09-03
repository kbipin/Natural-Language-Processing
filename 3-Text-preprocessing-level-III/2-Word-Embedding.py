# -*- coding: utf-8 -*-
"""
@author: KUMAR BIPIN
@website : www.kumar-bipin.com
"""
# Word Embedding Techniques using Embedding Layer in Keras
from tensorflow.keras.preprocessing.text import one_hot

### The Example Sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

print(sent)

### Define the Vocabulary size
voc_size = 10000

### One Hot Representation
onehot_repr=[one_hot(words,voc_size)for words in sent] 
print(onehot_repr)

### Word Embedding Represntation

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential


### Define the Max Sentecne Length
sent_length = 8
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)

### Define the Dimention
dim = 10
model=Sequential()
model.add(Embedding(voc_size, dim, input_length=sent_length))
model.compile('adam','mse')

model.summary()

### predict the 1 Senetecne in the list
print(model.predict(embedded_docs)[0])

### predict the list of Senetecnes in the list
print(model.predict(embedded_docs))
