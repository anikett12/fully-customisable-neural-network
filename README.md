# fully-customisable-neural-network
 This is the implementation of a fully customizable neural network with arbitrary no. of hidden layers using only NumPy, in Python.
 
 
 #Documentation
 - tanh activation function has been used in this model.
 - ```tanhderiv(K)```: calculates the derivative of the tanh function.
 - ```initialisetheta(m,n,nodes,yn)```: initialises the parameter theta for all the given layers.
 - ```hiddenlayer(m,nodes)```: initialises the hidden layers with one in all nodes.
 - ```forwardpropogation(x,theta,hidden)``` : performs a pass of forward propogation on X and returns the output as well as the updated
                                              value of hidden layers.
 - ```delta_function(m,hidden,output,y,theta)``` : Calculates derivative of  loss with respect to activations in each layer.
 - ``` update_theta(x,delta,hidden,theta)``` : Updates the value of weights for a single pass of forward and backward propogation.
 - ```main()``` : performs the forward and backward propogation a given number of times to minimise the loss function. 
