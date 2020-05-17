Convolutional Online Adaptation Learning
===
This code implements the model discussed in Convolutional Online Adaptation Learning for Opinion Mining. Content is continously posted online from different domains. Hence, we propose a model that can be trained in one domain and predict sentiments in another domain. We consider recurrent neural networks to rememeber te label of the previous sentence and online learning that can update parameters with a single sentence. A sentiment prior based on bi-gram word vector is used during online learning. The model is evaluated on Amazon product reviewes from different domains such as 'Electornics' and 'Books'.

Requirements
---
This code is based on the Online SVM code found at:
https://github.com/Wotipati/Passive_Aggressive

Convolutional Neural Networks 
---

This code is based on supervised convolutional neural networks are found here :
http://deeplearning.net/tutorial/lenet.html

We have extended the model to allow unsupervised pre-training.

Training
---
Train the model:
python convolutional_mlp.py

 - The training data will be taken from 'training.pkl.gz'
 - The tri-gram features are in the 'outputs' folder

Recurrent Neural Networks
---

Online SVM
---

