import rbms
import stats
import updaters
import trainers
import monitors
import units
import parameters
import base

import theano
import theano.tensor as T
import csv
import numpy as np
import gzip, time

import matplotlib.pyplot as plt
plt.ion()

from utils import generate_data, get_context

# DEBUGGING

mode = None

#############################################################################

import os
os.environ["THEANO_FLAGS"] = "base_compiledir=./theano/"
import sys
import timeit
import pickle
import numpy
import numpy.core.multiarray
import csv

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import pool

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

neuron = 10;
lay2w = 0;
all_test = 0;
eval_print1 = 0;
eval_print2 = 0;
eval_print3 = 0;
epoch_cd = 0;
batchm = 0;
indk = -1;

def morbrun1(f1=1, f2=1, v1=1, v2=1, kern = 1):
      
  test_set_x = np.array(eval_print1).flatten(order='C')
  valid_set_x = np.array(eval_print3).flatten(order='C')
  train_set_x = np.array(eval_print2).flatten(order='C')

  train_set_x = train_set_x.reshape(np.array(eval_print2).shape[0]*batchm,kern,v1,v2)
  valid_set_x = valid_set_x.reshape(np.array(eval_print3).shape[0]*batchm,kern,v1,v2)   
  test_set_x = test_set_x.reshape(np.array(eval_print1).shape[0]*batchm,kern,v1,v2)

  visible_maps = kern
  hidden_maps = neuron 
  filter_height = f1 
  filter_width = f2
  mb_size = batchm # 1 minibatch
  
  print(">> Constructing RBM...")
  fan_in = visible_maps * filter_height * filter_width

  """
   initial_W = numpy.asarray(
            self.numpy_rng.uniform(
                low = - numpy.sqrt(3./fan_in),
                high = numpy.sqrt(3./fan_in),
                size = self.filter_shape
            ), dtype=theano.config.floatX)
  """
  numpy_rng = np.random.RandomState(123)
  initial_W = np.asarray(
            numpy_rng.normal(
                0, 0.5 / np.sqrt(fan_in),
                size = (hidden_maps, visible_maps, filter_height, filter_width)
            ), dtype=theano.config.floatX)
  initial_bv = np.zeros(visible_maps, dtype = theano.config.floatX)
  initial_bh = np.zeros(hidden_maps, dtype = theano.config.floatX)



  shape_info = {
   'hidden_maps': hidden_maps,
   'visible_maps': visible_maps,
   'filter_height': filter_height,
   'filter_width': filter_width,
   'visible_height': v1, #45+8,
   'visible_width': v2, #30,
   'mb_size': mb_size
  }

  # rbms.SigmoidBinaryRBM(n_visible, n_hidden)
  rbm = base.RBM()
  rbm.v = units.BinaryUnits(rbm, name='v') # visibles
  rbm.h = units.BinaryUnits(rbm, name='h') # hiddens
  rbm.W = parameters.Convolutional2DParameters(rbm, [rbm.v, rbm.h], theano.shared(value=initial_W, name='W'), name='W', shape_info=shape_info)
  # one bias per map (so shared across width and height):
  rbm.bv = parameters.SharedBiasParameters(rbm, rbm.v, 3, 2, theano.shared(value=initial_bv, name='bv'), name='bv')
  rbm.bh = parameters.SharedBiasParameters(rbm, rbm.h, 3, 2, theano.shared(value=initial_bh, name='bh'), name='bh')

  initial_vmap = { rbm.v: T.tensor4('v') }

  # try to calculate weight updates using CD-1 stats
  print(">> Constructing contrastive divergence updaters...")
  s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=5, mean_field_for_stats=[rbm.v], mean_field_for_gibbs=[rbm.v])


  lr_cd = 0.001
  if indk == -1:
      lr_cd = 0
  
  umap = {}
  for var in rbm.variables:
    pu =  var + lr_cd * updaters.CDUpdater(rbm, var, s)
    umap[var] = pu

  print(">> Compiling functions...")
  t = trainers.MinibatchTrainer(rbm, umap)
  m = monitors.reconstruction_mse(s, rbm.v)

  e_data = rbm.energy(s['data']).mean()
  e_model = rbm.energy(s['model']).mean()


  # train = t.compile_function(initial_vmap, mb_size=32, monitors=[m], name='train', mode=mode)
  train = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[m, e_data, e_model], name='train', mode=mode)


  # TRAINING 

  epochs = epoch_cd
  print(">> Training for %d epochs..." % epochs)



  for epoch in range(epochs):
    monitoring_data_train = [(cost, energy_data, energy_model) for cost, energy_data, energy_model in train({ rbm.v: train_set_x })]
    mses_train, edata_train_list, emodel_train_list = zip(*monitoring_data_train)
  
  
  lay1w = rbm.W.var.get_value()
  Wl = theano.shared(lay1w) 
  lay1bh = rbm.bh.var.get_value() 
  bhl = theano.shared(lay1bh)
  return [Wl, bhl]
 
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(1, 1), W=None, b=None, bmode='valid'):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
                   
        # initialize weights with random weights
        if self.W is None:
          W_bound = numpy.sqrt(6. / (fan_in + fan_out))
          self.W = theano.shared(
             numpy.asarray(
                 rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                 dtype=theano.config.floatX
             ),
             borrow=True
           )

        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
          b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
          self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            border_mode = bmode 
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def evaluate_lenet5(learning_rate=0.05, n_epochs=100,
                    dataset='training.pkl.gz',
                    nkerns=[5, 5, 5, 5, 5, 5, 5, 5, 5], batch_size=50, dirn='iti', indexd=0):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    global layer0gW
    global layer1gW
    global layer1bgW
    global layer2gW
    global layer3gW
    global layer0gb
    global layer1gb
    global layer1bgb
    global layer2gb
    global layer3gb
    global all_test
    global batchm  
    global eval_print1
    global eval_print2
    global eval_print3
    global neuron
    global epoch_cd
    global indk
    
    epoch_cd = 10
    neuron = 5;  
    batchm  = 20
    batch_size = batchm
    
    for nk in range(9):
        nkerns[nk]=neuron
      
    dirgtest = dirn;

    l_r = T.scalar('l_r', dtype=theano.config.floatX)
                  
    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2] 
     
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
                                                                                                                                                                                                                 
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    n_train_batches = int(n_train_batches)
    n_valid_batches = int(n_valid_batches)
    n_test_batches = int(n_test_batches)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # Each sentence is of maximum length 73 words
    # Each word has word-vector length 5
    # 8 words are used for padding on both ends of sentence

    im1x = 73+8
    im1y =  5
    
    layer0_input = x.reshape((batch_size, 1, im1x, im1y))

    # Construct the first convolutional layer:
    # filtering reduces the image size to ((73+8)-3+1 , 5-5+1) = (78, 1)
    
    nk1x=3
    nk1y=im1y
                                                                                      
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, im1x, im1y),
        filter_shape=(nkerns[0], 1, nk1x, nk1y),
    )

    # Construct the second convolutional layer
    # filtering reduces the image size to (78-3+1, 1-1+1) = (76, 1)
   
    im2x = (im1x-nk1x+1)
    im2y = (im1y-nk1y+1)
    nk2x=6
    nk2y=im2y
    
    
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], im2x, im2y),
        filter_shape=(nkerns[1], nkerns[0], nk2x, nk2y),
      
    )

    # Construct the third convolutional layer
    # filtering reduces the image size to (76-3+1, 1-1+1) = (74, 1)
    
    im2bx = (im2x-nk2x+1)
    im2by = (im2y-nk2y+1)
    nk2bx=3
    nk2by=im2by         
    
    layer1b = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], im2bx, im2by),
        filter_shape=(nkerns[2], nkerns[1], nk2bx, nk2by),
       
    )

    # Construct the fourth convolutional layer
    # filtering reduces the image size to (74-3+1, 1-1+1) = (72, 1)

    im2cx = (im2bx-nk2bx+1)
    im2cy = (im2by-nk2by+1)
    nk2cx=4
    nk2cy=im2cy
     
   
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),


    layer2_input = layer1b.output.flatten(2)
    im3x = (im2bx-nk2bx+1)
    im3y = (im2by-nk2by+1)
 


    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[4] * im3x * im3y,
        n_out=100,
        activation=T.tanh
      
    )

    
                  
    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=100, n_out=2)
    
    # the cost we minimize during training is the NLL of the model
   
    cost = layer3.negative_log_likelihood(y)
    
    
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    test_model2 = theano.function(
        [index],
        layer3.errors2(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1b.params + layer1.params + layer0.params 
   
    if indexd > indk:
        
        epoch_cd = 1
        learning_rate=0
        n_epochs = 1       
        f = open(dirgtest+"/weights/layer0w_"+str(indexd)+".save",'rb')
        lay_params = pickle.load(f)
        Wl1, bl1 = lay_params
        layer0.W.set_value(Wl1.get_value());
        layer0.b.set_value(bl1.get_value());
        f.close()         
        f = open(dirgtest+"/weights/layer1w_"+str(indexd)+".save",'rb')
        lay_params = pickle.load(f)
        Wl1, bl1 = lay_params
        layer1.W.set_value(Wl1.get_value());
        layer1.b.set_value(bl1.get_value());
        f.close()
        f = open(dirgtest+"/weights/layer1bw_"+str(indexd)+".save",'rb')
        lay_params = pickle.load(f)
        Wl1, bl1 = lay_params
        layer1b.W.set_value(Wl1.get_value());
        layer1b.b.set_value(bl1.get_value());
        f.close()
        f = open(dirgtest+"/weights/layer2w_"+str(indexd)+".save",'rb')
        lay_params = pickle.load(f)
        Wl1, bl1 = lay_params
        layer2.W.set_value(Wl1.get_value());
        layer2.b.set_value(bl1.get_value());
        f.close()
        f = open(dirgtest+"/weights/layer3w_"+str(indexd)+".save",'rb')
        lay_params = pickle.load(f)
        Wl1, bl1 = lay_params
        layer3.W.set_value(Wl1.get_value());
        layer3.b.set_value(bl1.get_value());
        f.close()

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - l_r * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
  
    train_model = theano.function(
        [index, l_r],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
                               
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        learning_rate = 0.99*learning_rate
        if epoch == 1:
             
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer0_input,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in range(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer0_input,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in range(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer0_input,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in range(n_valid_batches)
                 ]
             
             if indk == 10:                      
              Wl1, bl1 = morbrun1(nk1x,nk1y,im1x,im1y)
              learning_rate = 0;
                
                
        if epoch == 2:
            
             layer0.W.set_value(Wl1.get_value());
             layer0.b.set_value(bl1.get_value());
             
             
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer0.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in range(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer0.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in range(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer0.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in range(n_valid_batches)
                 ]  
             
             if indk == 10:        
              Wl2, bl2 = morbrun1(nk2x,nk2y,im2x,im2y,neuron)
          
        
        if epoch == 3:
             layer1.W.set_value(Wl2.get_value());
             layer1.b.set_value(bl2.get_value());
             layer0.W.set_value(Wl1.get_value());
             layer0.b.set_value(bl1.get_value());
             
             
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer1.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in range(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer1.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in range(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer1.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in range(n_valid_batches)
                 ]  
             
             if indk == 10:        
              Wl3, bl3 = morbrun1(nk2bx,nk2by,im2bx,im2by,neuron)
              
              layer1b.W.set_value(Wl3.get_value());
              layer1b.b.set_value(bl3.get_value());
              layer1.W.set_value(Wl2.get_value());
              layer1.b.set_value(bl2.get_value());
              layer0.W.set_value(Wl1.get_value());
              layer0.b.set_value(bl1.get_value());
  
              n_in=nkerns[4] * im3x * im3y
              n_out=100
              W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
              )
              layer2.W.set_value(W_values);
              layer2.b.set_value(numpy.zeros(n_out))
              n_in=100
              n_out=2
              W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
              )
              layer3.W.set_value(W_values);
              layer3.b.set_value(numpy.zeros(n_out))
              learning_rate=0.01
         
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            
                    

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index, learning_rate)
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    
                    test_score = numpy.mean(test_losses)
                    
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break


    end_time = timeit.default_timer()
  
    try:
          os.remove(dirgtest+"/outputs/pred_y"+str(indexd)+".csv")
    except OSError:
     pass   
         
    predy=open(dirgtest+"/outputs/pred_y"+str(indexd)+".csv",'a')   

    test_losses = [
                        test_model2(i)
                        for i in range(n_test_batches)
                    ]
                 
                       
    np.savetxt(predy,test_losses,delimiter='\n');
    
    predy.close();
    
    
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    all_test += test_score
    
    with open(dirgtest+"/outputs/cv_score.txt", "a") as myfile:
          myfile.write(str(test_score)+"\n")
         
    if indk == 10: 
        
        eval_set_x = test_set_x;
        eval_shape = train_set_x.get_value(borrow=True).shape;

        eval_layer2 = theano.function([index], layer2.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

        eval_print = [
                   eval_layer2(i)
                   for i in range(n_test_batches)
                 ]
        myfile = open("outputs/layer0_vid_test"+str(indexd)+".csv",'w');
        wr = csv.writer(myfile, quotechar=None,escapechar='\\')
        wr.writerows(eval_print)
        myfile.close()

        eval_set_x = train_set_x;
        eval_shape = train_set_x.get_value(borrow=True).shape;

        eval_layer2 = theano.function([index], layer2.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

        eval_print = [
                   eval_layer2(i)
                   for i in range(n_train_batches)
                 ]

        myfile = open("outputs/layer0_vid_train"+str(indexd)+".csv",'w');
        wr = csv.writer(myfile, quotechar=None,escapechar='\\')
        wr.writerows(eval_print)
        myfile.close()

        eval_set_x = valid_set_x;
        eval_shape = train_set_x.get_value(borrow=True).shape;

        eval_layer2 = theano.function([index], layer2.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

        eval_print = [
                   eval_layer2(i)
                   for i in range(n_valid_batches)
                 ]

        myfile = open("outputs/layer0_vid_val"+str(indexd)+".csv",'w');
        wr = csv.writer(myfile, quotechar=None,escapechar='\\')
        wr.writerows(eval_print)
        myfile.close()   
        
    if indk == 10:      
        
        print("saving \n");    
        
        f = open(dirgtest+"/weights/layer0w_"+str(indexd)+".save", 'wb')
        pickle.dump(layer0.params, f)  
        f.close()    
        
        f = open(dirgtest+"/weights/layer1w_"+str(indexd)+".save", 'wb')
        pickle.dump(layer1.params, f)  
        f.close()   
        
        f = open(dirgtest+"/weights/layer1bw_"+str(indexd)+".save", 'wb')
        pickle.dump(layer1b.params, f)  
        f.close()                                                                          
        
        f = open(dirgtest+"/weights/layer2w_"+str(indexd)+".save", 'wb')
        pickle.dump(layer2.params, f)  
        f.close() 
        
        f = open(dirgtest+"/weights/layer3w_"+str(indexd)+".save", 'wb')
        pickle.dump(layer3.params, f)  
        f.close() 
     

if __name__ == '__main__':
    dirgtest = '.'
    
    try:
       os.remove(dirgtest+"cv_score.txt")
    except OSError:
        pass
    
    indk = -1;  # 10 for training and -1 for testing
    for i in range(1):     
          nameg = 'training.pkl.gz'
          evaluate_lenet5(dataset=nameg, n_epochs = 5, dirn=dirgtest, indexd=i)
        
def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
