from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        y_pred = X[i].dot(W)
        y_exp = np.exp(y_pred)
        y_exp_sum = np.sum(y_exp)
        softmax_correct_y = y_exp[y[i]] / y_exp_sum
        loss += -np.log(softmax_correct_y) # one hot cross entropy loss
        
        dW[:, y[i]] +=   X[i]*(softmax_correct_y-1)
        for j in range(num_classes):
            if not j == y[i]:
                dW[:, j] += X[i]*(y_exp[j])/(y_exp_sum)
        
    loss /= num_train
    loss += reg*np.sum(W*W)
    
    dW /= num_train
    dW += reg*2*W
            

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    y_pred = X.dot(W)
    y_exp = np.exp(y_pred)
    y_exp_sum = np.sum(y_exp,axis=1)
    softmax_y = y_exp/np.expand_dims(y_exp_sum,1)   # y_exp[np.arange(num_train),y] / y_exp_sum
    softmax_correct_y = softmax_y[np.arange(num_train),y]
    loss = np.sum(-np.log(softmax_correct_y))
    
    loss /= num_train
    loss += reg*np.sum(W*W)
    
    onehot_y = np.zeros_like(y_pred)
    onehot_y[np.arange(num_train),y] = 1
    dW = X.T.dot(softmax_y-onehot_y)
    
    dW /= num_train
    dW += reg*2*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
