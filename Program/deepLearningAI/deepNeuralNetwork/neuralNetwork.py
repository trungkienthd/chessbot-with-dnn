import pickle

import numpy as np

from deepLearningAI.deepNeuralNetwork.layers import StandardLayer, BinaryCrossEntropy

class NeuralNetwork():
    
    def __init__(self, inputDimension, hiddenDimensions=[], outputDimension=1, learningRate=3e-6, regularization=0.01, activationFunctions=[None]):
        
        np.set_printoptions(threshold=np.inf)
        
        self.dimensions = [inputDimension] + hiddenDimensions
        
        if activationFunctions is None:
            activationFunctions = ["ReLU"] * (len(hiddenDimensions)) + ["Sigmoid"]
        elif len(activationFunctions) < len(hiddenDimensions):
            activationFunctions += ["ReLU"] * (len(hiddenDimensions) - len(activationFunctions)) + ["Sigmoid"]
        
        # Initialize the hidden layers
        self.hiddenLayers = []
        for i in range(0, len(hiddenDimensions)):
            self.hiddenLayers.append(StandardLayer(inputDimension=self.dimensions[i], outputDimension=self.dimensions[i + 1], learningRate=learningRate, regularization=regularization, activationFunction=activationFunctions[i]))
            
        # Initialize the final layer
        self.finalLayer = StandardLayer(inputDimension=self.dimensions[-1], outputDimension=outputDimension, learningRate=learningRate, regularization=regularization, activationFunction=activationFunctions[-1])
        
        # Initialize the cost function
        self.costFunction = BinaryCrossEntropy()
        
        self.printNeuralNetwork()
        
    def printNeuralNetwork(self):
        print("\n=======================================================================================")
        print("Neural network's components:")
        print(" - An Input layer, taken {} input features".format(self.dimensions[0]))
        print(" - {} Hidden layers".format(len(self.hiddenLayers)))
        for i in range(0, len(self.hiddenLayers)):
            print("     + Hidden layer {}: {} units, with {} activation function".format(i + 1, self.hiddenLayers[i].linearLayer.numberOfUnits, self.hiddenLayers[i].activationFunctionLayer))
            print("         Weights: {} \n{}".format(self.hiddenLayers[i].linearLayer.W.shape, self.hiddenLayers[i].linearLayer.W))
            print("         Biases: {} \n{}".format(self.hiddenLayers[i].linearLayer.b.shape, self.hiddenLayers[i].linearLayer.b))
        print(" - An Output layer with {} units - {} activation function\n".format(self.finalLayer.linearLayer.numberOfUnits, self.finalLayer.activationFunctionLayer))
        print("     Weights: {} \n{}".format(self.finalLayer.linearLayer.W.shape, self.finalLayer.linearLayer.W))
        print("     Biases: {} \n{}".format(self.finalLayer.linearLayer.b.shape, self.finalLayer.linearLayer.b))
      
    # def updateLearningRate(self, value):
    #     for layer in self.hiddenLayers:
    #         layer.linear.learningRate = value
            
    #     self.finalLayer.learningRate = value
        
    def predict(self, X):
        
        # Forward pass through the hidden layers
        hidden = X
        for layer in self.hiddenLayers:
            hidden = layer.forward(hidden)
            
            # print("\n========================================================================")
            # print("Hidden")
            # print("     Shape: {}".format(hidden.shape))
            # print("     Values: {}".format(hidden))
            
        # scores is the output of the forward propagation (prediction values)
        scores = self.finalLayer.forward(hidden)
        
        # print("\n========================================================================")
        # print("Scores")
        # print("     Shape: {}".format(scores.shape))
        # print("     Values: {}".format(scores))
        
        return scores.flatten()
    
    def forward(self, X, labels, training=True):
        
        # Forward pass through the hidden layers
        hidden = X
        for layer in self.hiddenLayers:
            hidden = layer.forward(hidden)
            
            # print("\n========================================================================")
            # print("Hidden")
            # print("     Shape: {}".format(hidden.shape))
            # print("     Values: {}".format(hidden))
                
        # scores is the output of the forward propagation (prediction values)
        scores = self.finalLayer.forward(hidden)
        
        # print("\n========================================================================")
        # print("Scores")
        # print("     Shape: {}".format(scores.shape))
        # print("     Values: {}".format(scores))
        
        # Calculate the cost
        cost = self.costFunction.forward(scores=scores, labels=labels)
        
        if training:
            # If the process is training, the we apply the backpropagation:
            #   Calculate the gradient of the cost with respect to the scores
            deltaAL = self.costFunction.backward(labels=labels)
            
            #   Backward pass to the final layer
            (deltaAL, _, _) = self.finalLayer.backward(deltaAL)
            
            #   Backward pass to the remaining layers
            for i in range(len(self.hiddenLayers) - 1, -1, -1):
                (deltaAL, _, _) = self.hiddenLayers[i].backward(deltaAL)
            
        return scores, cost
    
    @staticmethod
    def save(model, modelFileName="model.pickle"):
        with open("./deepLearningAI/models/{}".format(modelFileName), "wb") as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(modelFileName="model.pickle"):
        with open("./deepLearningAI/models/{}".format(modelFileName), "rb") as file:
            model = pickle.load(file)
        return model