import numpy as np
import theano, cPickle, time, os
import theano.tensor as T
from theano.compat.python2x import OrderedDict
import pdb

from theano.tensor.shared_randomstreams import RandomStreams

def sharedX(x) : return theano.shared( theano._asarray(x, dtype=theano.config.floatX) ) 
def randn(shape,mean,std) : return sharedX( mean + std * np.random.standard_normal(size=shape) )
def zeros(shape) : return sharedX( np.zeros(shape) ) 

def sfmx(x) : return T.nnet.softmax(x)
def tanh(x) : return T.tanh(x)
def sigm(x) : return T.nnet.sigmoid(x)

def negative_log_likelihood(probs, labels) : # labels are not one-hot code 
    return - T.mean( T.log(probs)[T.arange(labels.shape[0]), T.cast(labels,'int32')] )

def predict(probs) : return T.argmax(probs, axis=1) # predict labels from probs
def error(pred_labels,labels) : return 100.*T.mean(T.neq(pred_labels, labels)) # get error (%)
def mse(x,y) : return T.sqr(x-y).sum(axis=1).mean() # mean squared error
def rand_ortho(shape, irange) : 
    A = - irange + 2 * irange * np.random.rand(*shape)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return sharedX(  np.dot(U, np.dot( np.eye(U.shape[1], V.shape[0]), V ))  )

def onehot(y, label_range):
    y_onehot = np.zeros([y.shape[0], label_range])
    y_onehot[np.arange(y.shape[0]), y.astype('int')] = 1
    return y_onehot

# load MNIST data into shared variables
(train_x, train_y), (_, _), (test_x, test_y) = np.load('/data/lisatmp3/saizheng/problems/targetprop/mnist.pkl')
train_y_onehot = onehot(train_y, 10)
test_y_onehot = onehot(test_y, 10)
num = 50000
num_t = 10000
train_x, train_y, test_x, test_y, train_y_onehot, test_y_onehot = \
    sharedX(train_x[:num,:]), sharedX(train_y[:num]), sharedX(test_x[:num_t,:]),  sharedX(test_y[:num_t]), sharedX(train_y_onehot[:num]), sharedX(test_y_onehot[:num_t])

#1 Hidden Layer, minibatch_size = 1
def exp1(__lr_vx, __lr_vy, __lr_wx, __lr_wy, seed = 27):
    max_epochs, batch_size, n_batches = 2000, 100, num/100

    nX, nH, nY = 784, 500, 10
    noise_level = 0.01
    W_x = rand_ortho((nX, nH), np.sqrt(6./(nX +nH)));  B_x = zeros((nH,))
    W_y = rand_ortho((nY, nH), np.sqrt(6./(nY +nH)));  B_y = zeros((nH,))    
    V_x = rand_ortho((nH, nX), np.sqrt(6./(nX +nH)));  C_x = zeros((nX,))
    V_y = rand_ortho((nH, nY), np.sqrt(6./(nY +nH)));  C_y = zeros((nY,))
    
    rng = RandomStreams(2 ** seed)
    
    Y_0 = T.vector() #not one-hot
    X, Y = T.matrix(), T.matrix()

    def sampleH_givenXY(x, y):
        h = T.switch(T.eq(rng.binomial((1,1), 1, 0.5)[0,0], 1),
                          sigm(T.dot(x + rng.normal(x.shape, 0, noise_level), W_x) + B_x),
                          sigm(T.dot(y + rng.normal(y.shape, 0, noise_level), W_y) + B_y))
        return h + rng.normal(h.shape, 0, noise_level)
    
    def sampleH_givenXYmean(x, y):
        #h = 0.5*(sigm(T.dot(x + rng.normal(x.shape, 0, noise_level), W_x) + B_x) +
        #    sigm(T.dot(y + rng.normal(y.shape, 0, noise_level), W_y) + B_y))
        
        h = sigm(T.dot(x + rng.normal(x.shape, 0, noise_level), W_x) + B_x +
                 T.dot(y + rng.normal(y.shape, 0, noise_level), W_y) + B_y)
        return h + rng.normal(h.shape, 0, noise_level)

    def sampleXY_givenH(h):
        x = sigm(T.dot(h + rng.normal(h.shape, 0, noise_level), V_x) + C_x)
        y = sigm(T.dot(h + rng.normal(h.shape, 0, noise_level), V_y) + C_y)
        return [x + rng.normal(x.shape, 0, noise_level), y + rng.normal(y.shape, 0, noise_level)]
 
    def predY_givenX(x):
        h = sigm(T.dot(x, W_x)+B_x)
        y = predict(sigm(T.dot(h, V_y) + C_y))
        return y

    #def sampleY_givenX(x):
    #    h = sigm(T.dot(x + rng.normal(x.shape, 0, noise_level), W_x) + B_x)
    #    y = sigm(T.dot(h, V_y) + C_y)
    #    return y

    # Denoising Autoencoder reconstruction error
    H_0 = sampleH_givenXYmean(X, Y) 
    H_0_noise = H_0 + rng.normal(H_0.shape, 0, noise_level) 
    X_rec_mean = sigm(T.dot(H_0_noise, V_x) + C_x)
    Y_rec_mean = sigm(T.dot(H_0_noise, V_y) + C_y)

    cost_X_rec = mse(X, X_rec_mean)
    cost_Y_rec = mse(Y, Y_rec_mean)

    #cost_H_rec = 0.5*mse(H_0, sigm(T.dot(X + rng.normal(X.shape, 0, noise_level), W_x)+B_x)) + \
    #             0.5*mse(H_0, sigm(T.dot(Y + rng.normal(Y.shape, 0, noise_level), W_y)+B_y))
    cost_H_rec = mse(H_0, sigm(T.dot(X + rng.normal(X.shape, 0, noise_level), W_x)+B_x + \
        T.dot(Y + rng.normal(Y.shape, 0, noise_level), W_y)+B_y))

    #cost_H_rec = mse(H_0, 0.5*(sigm(T.dot(X + rng.normal(X.shape, 0, noise_level), W_x) + B_x) +
    #           sigm(T.dot(Y + rng.normal(Y.shape, 0, noise_level), W_y) + B_y)))
    
    #Error which forces P_1(H|X,Y) and P_2(X,Y|H) to be consistent to unique joint P.
    X_1, Y_1 = sampleXY_givenH(H_0)
    H_1 = sampleH_givenXYmean(X, Y)
    H_1_noise = H_1 + rng.normal(H_0.shape, 0, noise_level)
    X_p_mean = sigm(T.dot(H_1_noise, V_x) + C_x) 
    Y_p_mean = sigm(T.dot(H_1_noise, V_y) + C_y) 

    cost_X_p = mse(X_1, X_p_mean)
    cost_Y_p = mse(Y_1, Y_p_mean)

    #cost_H_p = 0.5*mse(H_1, sigm(T.dot(X_1 + rng.normal(X_1.shape, 0, noise_level), W_x)+B_x)) + \
    #           0.5*mse(H_1, sigm(T.dot(Y_1 + rng.normal(Y_1.shape, 0, noise_level), W_y)+B_y))
    
    cost_H_p = mse(H_1, sigm(T.dot(X_1 + rng.normal(X_1.shape, 0, noise_level), W_x)+B_x + \
        T.dot(Y_1 + rng.normal(Y_1.shape, 0, noise_level), W_y)+B_y))

    #cost_H_p = mse(H_1, 0.5*(sigm(T.dot(X_1 + rng.normal(X_1.shape, 0, noise_level), W_x) + B_x) +
    #           sigm(T.dot(Y_1 + rng.normal(Y_1.shape, 0, noise_level), W_y) + B_y)))
    
    cost = cost_X_rec + cost_Y_rec + cost_H_rec + cost_X_p + cost_Y_p 
    err = error(predY_givenX(X), Y_0)

    r1 = 1;
    r2 = 1;
    gV_x, gC_x = T.grad(r1*cost_X_rec + r2*cost_X_p, [V_x, C_x], consider_constant = [H_0_noise, H_1_noise])
    gV_y, gC_y = T.grad(r1*cost_Y_rec + r2*cost_Y_p, [V_y, C_y], consider_constant = [H_0_noise, H_1_noise])
    gW_x, gB_x = T.grad(r1*cost_H_rec + r2*cost_H_p, [W_x, B_x], consider_constant = [H_0, H_1])
    gW_y, gB_y = T.grad(r1*cost_H_rec + r2*cost_H_p, [W_y, B_y], consider_constant = [H_0, H_1])

    """ Training"""
    i = T.lscalar();
    train_batch = theano.function( [i], [cost, err],
        givens={ X : train_x[ i*batch_size : (i+1)*batch_size ],
                 Y : train_y_onehot[ i*batch_size : (i+1)*batch_size ],
                 Y_0 : train_y[ i*batch_size : (i+1)*batch_size ]},
        updates=OrderedDict({
            V_x : V_x - __lr_vx*gV_x,  C_x : C_x - __lr_vx*gC_x,
            V_y : V_y - __lr_vy*gV_y,  C_y : C_y - __lr_vy*gC_y,
            W_x : W_x - __lr_wx*gW_x,  B_x : B_x - __lr_wx*gB_x,
            W_y : W_y - __lr_wy*gW_y,  B_y : B_y - __lr_wy*gB_y, 
            })  )
    
    eval_test = theano.function( [i], [err], # make evaluation function for test error
        givens={    X : test_x[ i*1000 : (i+1)*1000 ],
                    #Y : test_y_onehot[ i*1000 : (i+1)*1000 ],
                    Y_0 : test_y[ i*1000 : (i+1)*1000 ] }  )
    
    
    t = time.time(); monitor = { 'train' : [], 'test'  : [] }
    print "[epoch] cost = [ train_cost  train_err ] [test_err]  time(sec)"
    
    for e in range(1,max_epochs+1) :
        monitor['train'].append(  np.array([ train_batch(i) for i in range(n_batches) ]).mean(axis=0)  )
        #pdb.set_trace()    
        if e % 2 == 0 : # display cost and errors
            monitor['test'].append( np.array([ eval_test(i)  for i in range(10) ]).mean(axis=0)  )
            print "[%5d] cost =" % (e), monitor['train'][-1], monitor['test'][-1], " %8.2f sec" % (time.time() - t)

exp1(0.1, 0.1, 0.1, 0.1)
