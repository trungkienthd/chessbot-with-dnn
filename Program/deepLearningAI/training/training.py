import pandas as pd
import numpy as np

from deepLearningAI.deepNeuralNetwork.neuralNetwork import NeuralNetwork

class Trainer:
    
    def __init__(self, dataFileName, activationFunctionsOfHiddenLayers="ReLU", numberOfEpoches=10, batchSize=0):     
        
        self.dataFileName = dataFileName
        self.activationFunctionsOfHiddenLayers = activationFunctionsOfHiddenLayers
        
        self.dnn = NeuralNetwork(inputDimension=72, hiddenDimensions=[8], outputDimension=1, activationFunctions=[activationFunctionsOfHiddenLayers] * 1 + ["Sigmoid"])
        
        self.xTrain, self.yTrain, self.xTest, self.yTest = self.readData(dataFileName=dataFileName)
        
        self.batchSize = batchSize
        if self.batchSize == 0:
            self.trainWithStochasticGD(numberOfEpoches=numberOfEpoches)
        else:
            self.trainWithMiniBatchGD(numberOfEpoches=numberOfEpoches, batchSize=self.batchSize)
        
        self.test()
                
    def readData(self, dataFileName):
        data = pd.read_csv("./deepLearningAI/data/{}.csv".format(dataFileName))
        
        trainData = data.copy().sample(frac=0.8, random_state=42)
        testData = data.copy().drop(trainData.index)
        
        xTrain = trainData.iloc[:, :72].to_numpy()
        yTrain = trainData.iloc[:, -1].to_numpy()
        
        xTest = testData.iloc[:, :72].to_numpy()
        yTest = testData.iloc[:, -1].to_numpy()
        
        return xTrain.transpose(), yTrain, xTest.transpose(), yTest
    
    def trainWithStochasticGD(self, numberOfEpoches):
        
        print("\n========================================================================================================================")
        print("TRAINING: (with Stochastic Gradient Descend)")
        trainingCosts = []
        maes = []
        for epoch in range(0, numberOfEpoches):
            # Forward propagation
            scores, cost = self.dnn.forward(X=self.xTrain, labels=self.yTrain, training=True)
            
            # Calculate Mean Absolute Error (MAE)
            mae = np.mean(np.abs(scores - self.yTrain))
        
            print("\n     Epoch [{}/{}]; Cost: {}; MAE: {}".format(epoch + 1, numberOfEpoches, cost, mae))
            print("   -----------------------------------------------------------------------------------------------------------")
            
            trainingCosts.append(cost)
            maes.append(mae)
            
        trainingCosts = np.array(trainingCosts)
        maes = np.array(maes)
        
        NeuralNetwork.save(model=self.dnn, modelFileName="ModelDataFile-'{}'__HiddenLayersActivationFunction-'{}'__Epoches-'{}'__StochasticGD.pickle".format(self.dataFileName, self.activationFunctionsOfHiddenLayers, numberOfEpoches))
        
    def trainWithMiniBatchGD(self, numberOfEpoches, batchSize):
        
        print("\n========================================================================================================================")
        print("TRAINING: (with Mini-batch Gradient Descend - Batch Size: {})".format(batchSize))
        for epoch in range(0, numberOfEpoches):
            print("\n     Epoch [{}/{}]:".format(epoch + 1, numberOfEpoches))
            
            epochCosts = []
            epochMaes = []
            
            randomIndices = np.random.choice(self.xTrain.shape[1], size=self.xTrain.shape[1], replace=False)
            xShuffle = self.xTrain[:, randomIndices]
            yShuffle = self.yTrain[randomIndices]
            
            xBatches = np.array_split(xShuffle, batchSize, axis=1)
            yBatches = np.array_split(yShuffle, batchSize)
            
            scores = []
            
            for batch in range(0, min(len(xBatches), len(yBatches))):
                # Forward propagation
                batchScores, cost = self.dnn.forward(X=xBatches[batch], labels=yBatches[batch], training=True)
                
                # Calculate Mean Absolute Error (MAE)
                mae = np.mean(np.abs(batchScores - yBatches[batch]))
                
                print("\n       Batch [{}/{}]; Cost: {}; MAE: {}".format(batch + 1, min(len(xBatches), len(yBatches)), cost, mae))
                
                epochCosts.append(cost)
                epochMaes.append(mae)
                
                scores = np.append(scores, batchScores)
                
            averageCost = np.mean(epochCosts)
            averageMae = np.mean(epochMaes)
            
            print("\n     Epoch's Average Cost: {}; Epoch's Average MAE: {}".format(averageCost, averageMae))
            print("   -----------------------------------------------------------------------------------------------------------")
            
        NeuralNetwork.save(model=self.dnn, modelFileName="ModelDataFile-'{}'__HiddenLayersActivationFunction-'{}'__Epoches-'{}'__MiniBatchGD-'{}'.pickle".format(self.dataFileName, self.activationFunctionsOfHiddenLayers, numberOfEpoches, batchSize))
        
    def test(self):
        print("\n========================================================================================================================")
        print("TESTING:")
        scores = self.dnn.predict(X=self.xTest)
        print("     Scores: \n{}".format(scores))
        print("     Labels: \n{}".format(self.yTest))
        
        mae = np.mean(np.abs(scores - self.yTest))
        
        print("\n -> MAE: {}".format(mae))