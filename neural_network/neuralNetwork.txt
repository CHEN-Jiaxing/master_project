import numpy as np
import scipy.special
# neural network class definition
class neuralNetwork:
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes= inputnodes;
        self.hnodes= hiddennodes;
        self.onodes= outputnodes;

        # learning rate
        self.lr=learningrate;
        
        # link weight matrices, wih and who
        self.wih= (np.random.rand(self.hnodes, self.inodes)-0.5)
        self.who= (np.random.rand(self.onodes, self.hnodes)-0.5)
        
        # activation function is the sigmoid function
        self.activation_function= lambda x: scipy.special.expit(x);
        
        pass
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        input= np.array(inputs_list, ndmin=2).T
        targets= np.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs= np.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs= self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs= np.dot(self.who, hidden_outputs)
        # calculate signals emerging from final output layer
        final_outputs= self.activation_function(final_inputs)
        
        # output layer error is the target-actual
        output_errors= targets- final_outputs;
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_error= np.dot(self.who.T, output_errors);
        
        # update the weights for the links between the hiden and output layers
        self.who+= self.lr* np.dot((output_error*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs));
        
        # update the weights for the links between the input and hidden layers 
        self.whi+= self.lr* np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs));
        pass
    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs= np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs=np.dot(self.wih, inputs);
        # calculate signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs);
        
        # calculate signals into final output layer
        final_inputs=np.dot(self.who, hidden_outputs);
        # calculate signals emerging from final output layer
        final_outpus=self.activation_function(final_inputs);
        return final_outpus


# number of input, hidden and output nodes
input_nodes=3;
hidden_nodes=3;
output_nodes=3;

# learing  rate  is 0.5
learning_rate=0.5;

# craet instance of neural network
n=neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)