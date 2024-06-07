import pandas as pd
import numpy as np

from deepLearningAI.deepNeuralNetwork.neuralNetwork import NeuralNetwork

class Trainer:
    
    def __init__(self, dataFileName, activationFunctionsOfHiddenLayers="ReLU", numberOfEpoches=10):     
        
        self.dataFileName = dataFileName
        self.activationFunctionsOfHiddenLayers = activationFunctionsOfHiddenLayers
        
        self.dnn = NeuralNetwork(inputDimension=72, hiddenDimensions=[32, 16, 8, 4, 2], outputDimension=1, activationFunctions=[activationFunctionsOfHiddenLayers] * 5 + ["Sigmoid"])
        
        self.x, self.y = self.readData(dataFileName=dataFileName)
        
        self.train(numberOfEpoches=numberOfEpoches)
                
    def readData(self, dataFileName):
        data = pd.read_csv("./deepLearningAI/data/{}.csv".format(dataFileName))
        
        x = data.iloc[:, :72].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        
        return x.transpose(), y
    
    def train(self, numberOfEpoches):
        
        for epoch in range(0, numberOfEpoches):
            # Forward propagation
            cost = self.dnn.forward(X=self.x, labels=self.y, training=True)
        
            print("Epoch [{}/{}], Cost: {}".format(epoch + 1, numberOfEpoches, cost))
            print("\n========================================================================")
        
        NeuralNetwork.save(model=self.dnn, modelFileName="model_data_file_{}_hidden_layer_activation_functions{}_epoches_{}.pickle".format(self.dataFileName, self.activationFunctionsOfHiddenLayers, numberOfEpoches))
        