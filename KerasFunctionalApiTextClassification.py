
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
import collections
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')


df = pd.read_csv("/data/text/train_text.csv")


df['Comment'] = df['Comment'].map(lambda x:x.lower())
df['Comment'] = df['Comment'].str.replace('[^\w\s]','')
df['Comment'] = df['Comment'].str.replace(r'[\s]+', ' ')
df['Comment'] = df['Comment'].str.replace(r'[^A-Za-z0-9]+', ' ')
df['Comment'] = df['Comment'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


df.head(3)

X = df['Comment'].values
Y = df['Target']
Y_cat =to_categorical(Y)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
train_x_seq = tokenizer.texts_to_sequences(X)
word_count = len(tokenizer.word_counts)
maxlen_tr=max([len(i) for i in train_x_seq])


#padding the data
maxlen=maxlen_tr
train_x_pad =pad_sequences(train_x_seq, maxlen=maxlen)


#Create embedding layer
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Input,Activation,Flatten, Dropout
from keras.models import Sequential, Model

embedding_dim = 50

embedding_layer = Embedding(word_count+ 1,embedding_dim,input_length=maxlen,trainable=True)

sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Dense(100,activation="relu")(embedded_sequences)
x = Dropout(.2)(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Flatten()(x)
x = Dense(2,activation="softmax")(x)
model = Model(sequence_input,x)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(train_x_pad,Y_cat,epochs=1,batch_size=50,validation_split=0.3)


df1 = pd.read_csv("/data/text/test_text.csv")
#lower the text, Punctuation removal , Non Aplhanumeric character , Stopword
df1['Comment'] = df['Comment'].map(lambda x:x.lower())
df1['Comment'] = df['Comment'].str.replace('[^\w\s]','')
df1['Comment'] = df['Comment'].str.replace(r'[\s]+', ' ')
df1['Comment'] = df['Comment'].str.replace(r'[^A-Za-z0-9]+', ' ')
df1['Comment'] = df['Comment'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

X1 = df1['Comment'].values


test_x_seq = tokenizer.texts_to_sequences(X1)
test_x_pad =pad_sequences(test_x_seq, maxlen=maxlen)

test_predict=model.predict(test_x_pad)

out_prob=pd.DataFrame(test_predict, columns=[0,1])
out_id=pd.DataFrame({'id':df1['id'].values})
out_df=pd.concat([out_id, out_prob],axis=1)
out_df.head(3)






