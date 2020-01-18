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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
      # (1, D) dot (D, C) = (1, C)
      scores = X[i].dot(W)
      # Calculate softmax, stablized it by shifting max(score)
      prob = np.exp(scores-np.max(scores))/np.sum(np.exp(scores-np.max(scores)))
      # Look at the probability of true class, calculate loglikehood and add it to loss
      loss += -np.log(prob[y[i]])
      # Calculate gradients, for true class move it to opposite direction
      for j in xrange(num_classes):
          dW[:,j] += prob[j] * X[i].T
      dW[:, y[i]] -= X[i].T

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # (N, D) dot (D, C) = (N, C), stableize the softmax
  scores = X.dot(W)
  scores -= np.max(scores , axis=1).reshape(-1, 1)
  # (N, C)
  nprob = np.exp(scores)/np.sum(np.exp(scores), axis=-1, keepdims=True)
  # sum over (N, 1)
  loss = np.sum(-np.log(nprob[range(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W * W)
  # Prepare to calculate gradient, this is the gradient of softmax outputt
  nprob[np.arange(num_train), y] -= 1.0
  # (C, N) dot (N, D) = (C, D)
  dW = nprob.T.dot(X).T
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
