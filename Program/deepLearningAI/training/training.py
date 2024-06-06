from deepLearningAI.deepNeuralNetwork.neuralNetwork import NeuralNetwork

class Trainer:
    
    def __init__(self):        
        self.dnn = NeuralNetwork(inputDimension=72, hiddenDimensions=[36, 18, 9], outputDimension=1)
                
            