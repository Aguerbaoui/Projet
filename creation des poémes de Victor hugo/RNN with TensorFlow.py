# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:20:28 2019

@author: Nadia
"""


import tensorflow as tf
import numpy as np

#be sure to used tensorflow 2.0
#assert hasattr(tf,"function")#be sure to use tensorfLow 2.0
#open and process daataset
#you can used your own dataset with english text 
with open("rnn_dataset/victorhugo.txt","r") as f:
    text=f.read()
print(len(text))
print(text[:1000])
#afficher les 1000 caractéres
#pour s'inspirer des poemes du voctor hugo
#import unicode
#text=unicode.unicode(text)
text=text.lower()
text=text.replace("2","")
text=text.replace("1","")
text=text.replace("8","")
text=text.replace("5","")
text=text.replace("<","")
text=text.replace(">","")
text=text.replace("!","")
text=text.replace("-","")
text=text.replace("$","")
text=text.replace(";","")
text=text.strip()
vocab=set(text)
print(len(vocab),vocab)
print(text[:100])
#l'etape hedhii on va assurer le matching enyre chaque lettre el les nombres 
#Map each letter int 
#kol lettre ===> un nomber,benessba lel reseau de neurones les lettres n'ont pas de sens n'ont pas de signification
vocab_size=len(vocab)
vocab_to_int={l:i for i in enumerate(vocab)}
int_to_vacab={i:l for i ,l in enumerate(vocab)}
print("vocab_to_int",vocab_to_int)#un dictionnaire fiih A:0 par exemple
print()
print("int_to_vocab",int_to_vocab)#un dictionnaire fih 0:A
#afficher le text en des nombres codé c ad on va afficher des nombres text codé 
encoded=[vocab_to_int[l]for l in text]
encoded_sentence=encoded[:100]
print(encoded_sentence)
#afficher les lettres deja codé affichage des lettres
decoded_sentence=[int_to_vocab[i]for i in encoded_sentence]
print(decoded_sentence)
#les lettres hedhom naamloulhoom des jioin nlemhouhom maa baadhhhom
decoded_sentence="".join(decoded_sentence)
print(decoded_sentence)
##""""""""********************* video sample batch************** video 20
#########"##vidéo 21 ##########"
vocab_size=len(vocab)
####create the layers
#set the input of the modle
tf_inputs=tf.keras.input(shape=(None,),batch_size=64)
#@## y'apaa de subclassing
#convert each value of the input into a one encoding vector
one_hot=oneHot(len(vocab))(tf_inputs)
print(one_hot)
#Stack LSTM cells \\\ définitions des cellules LSTM
rnn_layer1=tf.keras.layers.LSTM(128,return_sequences=True,Stateful=True)(one_hot)
#voiir une vidéo sur les cellules LSTM
#notre 2 éme cellule LSTM
rnn_layer2=tf.keras.layers.LSTM(128,return_sequences=True,Stateful=True)(rnn_layer1)
### dérniére sortie (hidden_Layer)
#create the outputs of the model
hidden_Layer=tf.keras.layers.Dense(128,activation="relu")
#recupérer la sortie c'est un réseau de (rnn_Layer2) neurones classiques
outputs=tf.keras.layers.Dense(vocab_size,activation="softmax")(hidden_Layer)
#c'est un vecteur de 34 valeur avec une activation softmax pour avoir des valeurs de probaaaa
#♥setup the model
model=tf.keras.Model(input=tf_input,outputs=outputs)
####Check if we can reset the RNN cells 
#star by resetting the cells of the RNN 
model.reset_states()
#Get one batch
batch_inputs,batch_targets=next(gen_batch(input,targets,50,64))
print(batch_input.shape)
########Make a first prediction
outputs=model.predict(batch_inputs)
first_prediction=outputs[0][0] ####  louwela la premier  ere sortie élement et la 1 ére sequence 
print(first_prediction)
##Reset the satets of the RNN states
model.reset_states()
#Make an other prediction to check the difference
outputs=model.predict(batch_inputs)
second_prediction=output[0][0]
#Check if both prediction are equal
assert(set(first_prediction)==set(second_prediction))
###############voilaaa la derniere vidéééoo Générer des poémes aléatoires
loss_object=tf.keras.losses.sparse_categorical_crossentropy()
optimizer=tf.keras.optimizers.Adam(lr=0.001)
#set some metrics to track the progress of the training
#loss
train_loss=tf.keras.metrics.Mean(name='train_loss')
#accuraccy
train_accuracy=tf.keras.metrics.sparse_top_k_categorical_accuracy(name='train_accuracy')
#set the train method and the predict method in graph mode
@tf.function
def train_step(inputs,tragets):
    with tf.GradientTape() as Tape:
        #Make a prediction on all the batch
        predictions=model(inputs)
    #Get the error /loss on these predictions 
    loss=loss_object(targets,predictions)
    #compute the gradient with respect to the loss
    gradients=tape.gradient(loss,model.trainable_variables)
    #change the weights of the model
    optimizer.apply_gradient(loss,model.trainable_variables)
    #the metrics are accumulate over time you don't need to average it yourself
    train_loss(loss)
    train_accuracy(targets, predictions)
    
    
    
    
@tf.function
def predict(inputs):
    #Make a prediction on all the batch
    predictions=model(inputs)
    return predictions
#Train the model
model.reset_states()
for epoch in range(4000):
    for batch_inputs,batch_targtets in gen_batch(inputs,targets,100,64,noise=13):
        train_step(batch_inputs,batch_targets)
        template='\rEpoch{} : Train Loss: {} train accuracy: {}' ####"affichage 
        print(template.format(epoche,train_loss.result(),train_accuracy.result()*100,end=""))
        model.reset_states()
import json
model.save("model_rnn.h5")
with open ("model_rnn_vocab_to_int","w") as f:
    f.write(json.dumps(vocab_to_int))
with open("model_rnn_int_to_vocab","w") as f:
    f.write(json.dumps(int_to_vocab))
#Generate some Text6
import random
model.reset_states()
# elle génére des sequences aleatoires
size_poetries=300
poetries=np.zeros((64,size_poetries,1))
sequences=np.zeros((64,100))
for b in range(64):
    rd=np.random.randint(0,len(inputs)-100)
    sequences[b]=inputs[rd:rd+100]
for i in range(size_poetries+1):
    if i>0:
        poetries[:,i-1,:]=sequences
        softmax=predict(sequences)
        #set the next seq
        
    sequences=np.zeros((64,1))
    for b in range(64):
        argsort=argsort[:-1]
        #select one of the strongest 4 proposals
        sequences[b]=argsort[0]
for b in range(64):
    sentence="".join([int_to_vocab[i[0]]for i in poetries[b]])
    print(sentence)
    print("\n====================================\n")

    
    
    
    
    






