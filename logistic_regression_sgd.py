#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2


#plt.rcParams["figure.figsize"] = [15,10]
# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eta = [0.5,0.3,0.1,0.05,0.01]



# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.

#X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.
#t = data[:,3]
# For plotting data
#class1 = np.where(t==0)
#X1 = X[class1]
#class2 = np.where(t==1)
#X2 = X[class2]

# Initialize w.


# Error values over all iterations.


DATA_FIG = 1

# Set up the slope-intercept figure
# SI_FIG = 2
# plt.figure(SI_FIG)
# plt.rcParams.update({'font.size': 15})
# plt.title('Separator in slope-intercept space')
# plt.xlabel('slope')
# plt.ylabel('intercept')
# plt.axis([-5, 5, -10, 0])
# error=0
plt.rcParams["figure.figsize"] = [15,10]

plt.figure()

#random shuffling
#for k in range (0,max_iter):
  #np.random.shuffle(X)  
for k in eta:
  w = np.array([0.1, 0, 0])
  error=np.zeros(max_iter)
  
  for iter in range (0, max_iter):
    np.random.shuffle(data)
    X=data[:,0:3]
    t=data[:,3]
    e_all=[]
    for i in range (0,len(X)):
      # Compute output using current w on all data X.
      y = sps.expit(np.dot(X[i,:],w))
      # y_all.append(y)
      # y_all2.append(1-y)
    
      # Gradient of the error, using Eqn 4.91
      grad_e = (1/len(X))*(y-t[i])*X[i].T
    
      # Update w, *subtracting* a step in the error derivative since we're minimizing
      w_old = w
      w = w - k*grad_e
    
      e = -np.mean(np.multiply(t[i],np.log(y)) + np.multiply((1-t[i]),np.log(1-y)))
      e_all.append(e)
    
      # Plot current separator and data.  Useful for interactive mode / debugging.
      # plt.figure(DATA_FIG)
      # plt.clf()
      # plt.plot(X1[:,0],X1[:,1],'b.')
      # plt.plot(X2[:,0],X2[:,1],'g.')
      # a2.draw_sep(w)
      # plt.axis([-5, 15, -10, 10])

  
      # Add next step of separator in m-b space.
      # plt.figure(SI_FIG)
      # a2.plot_mb(w,w_old)

    
      # Print some information.
      #I commented out below line, it might be useful, CHECK LATER!!!
      #print 'epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e, w.T)
  
      # Stop iterating if error doesn't change more than tol.
  
      #if np.absolute(e-e_all[i-1]) < tol:
      #break
    # e is the error, negative log-likelihood (Eqn 4.90)
  
    # Add this error to the end of error vector.
    error[iter]=np.mean(e_all)
    # avg_e=np.mean(e_all)
    # if iter>0:
    # if np.absolute() < tol:
      # break

  plt.plot(error, label='eta=%.2f' %(k))
  
plt.legend()
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.show()  
