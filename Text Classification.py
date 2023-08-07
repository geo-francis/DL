import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing import text,sequence
from keras import utils #"utils" is an abbreviation for "utilities ",used as a module or package name in programming to represent a collection of utility functions or tools. These utility functions are typically helper functions that provide commonly used functionality and assist in performing various tasks
df=pd.read_csv(r"E:\Project\consumer_complaints.csv")
df.columns
col=["consumer_complaint_narrative","product"]
df_new=df[col]
df_new1=df_new[pd.notnull(df_new["consumer_complaint_narrative"])]
df_new1["product"].value_counts()
train_size=int(len(df_new1)*.8)
test_size=len(df_new1)-train_size
train_narrative=df_new1["consumer_complaint_narrative"][:train_size]
train_product=df_new1["product"][:train_size]
test_narrative=df_new1["consumer_complaint_narrative"][train_size:]
test_product=df_new1["product"][train_size:]
x=df_new1["consumer_complaint_narrative"]
y=df_new1["product"]
X_train_old,X_test_old,y_train_old,y_test_old=train_test_split(x,y,test_size=0.2,random_state=0)
max_words=1000
tokensize=text.Tokenizer(num_words=max_words,char_level=False)#The purpose of a tokenizer is to convert text data into a numerical representation that can be used for machine learning or natural language processing tasks.
tokensize.fit_on_texts(X_train_old)
X_train=tokensize.texts_to_matrix(X_train_old)
X_test=tokensize.texts_to_matrix(X_test_old)
encoder=LabelEncoder()
encoder.fit(y_train_old)## convert label to number (string -number)
y_train=encoder.transform(y_train_old)
y_test=encoder.transform(y_test_old)
num_classes=np.max(y_train)+1
y_train=utils.to_categorical(y_train,num_classes)
y_test=utils.to_categorical(y_test,num_classes)
##model
model=Sequential()## tell the process
model.add(Dense(512,input_shape=(max_words,)))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=4,verbose=1,validation_split=0.1)

SCORE=model.evaluate(X_test,y_test,batch_size=32,verbose=1)

SCORE
SCORE[0]
SCORE[1]
text_labels=encoder.classes_
for i in range(10):
    prediction=model.predict(np.array([X_test[i]]))
    predicted_label=text_labels[np.argmax(prediction)]
    print(X_test_old.iloc[i][:50],".....")
    print("actual label:"+ y_test_old.iloc[i])
    print("predcited label:"+ predicted_label)
    