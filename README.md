Online Sentiment Classification of Text
===
This code implements the model discussed in the paper _Convolutional Online Adaptation Learning for Opinion Mining_. Content is continously posted online from different domains. Hence, we propose a model that can be trained in one domain and predict sentiments in another domain. 

Requirements
---
This code is based on the Online SVM code found at:
https://github.com/Wotipati/Passive_Aggressive

Convolutional Recurrent Neural Networks 
---
We consider recurrent neural networks to rememeber te label of the previous sentence and online learning that can update parameters with a single sentence. We have allowed unsupervised pre-training.

Training
---
Train the CNN:
python convolutional_mlp.py

 - The training data will be taken from 'training.pkl.gz'
 - The tri-gram features are in the 'outputs' folder

Train the RNN:
matlab -m crnn

 - The training data will be taken from 'set_train_rnn0'
 - The output features are 'data_svm0.txt'


Online SVM
---
A sentiment prior based on bi-gram word vector is used during online learning. The model is evaluated on Amazon product reviewes from different domains such as 'Electornics' and 'Books'.

Train the SVM:

la_svm -g 0.005 -c 1 ../format/train.lib

Test the SVM:

python csv2lib.py test test.lib 0 False

ICDM 2020 Sentire
---
Paper link : https://github.com/ichaturvedi/convolutional-online-adaptation-learning/blob/master/iti-COAL.pdf

Presentation : https://youtu.be/vmCG3tjs7sQ
