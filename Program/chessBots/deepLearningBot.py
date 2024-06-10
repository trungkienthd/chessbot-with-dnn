import sys

import pandas as pd

from chessBots.bot import Bot
from deepLearningAI.deepNeuralNetwork.neuralNetwork import NeuralNetwork
from deepLearningAI.training.dataGenerating import ChessStateEncoder, DataGenerator

class DeepLearningBot(Bot):
    
    def __init__(self, chess=None, playerIndex=1, modelFileName="ModelDataFile-'1_Simulations_Of_White_RandomBot_VS_Black_RandomBot'__HiddenLayersActivationFunction-'ReLU'__Epoches-'150'__MiniBatchGD-'64'.pickle"):
        super().__init__(chess=chess)
        
        self.thinkingTime = 1.
        
        self.dnn = NeuralNetwork.load(modelFileName=modelFileName)
        self.dnn.printNeuralNetwork()
        
        self.playerIndex = playerIndex
        
        self.simulatedChessStateEncoder = ChessStateEncoder(self.chess.board)
        
    def __str__(self):
        return "DeepLearning"
    
    def perform(self):
        move = self.evaluatePossibleMoves()
        try:
            self.chess.makeAMove(moveToString=move)
        except:
            randomMove = random.choice(self.chess.getPossibleMoves())
            self.chess.makeAMove(moveToString=randomMove)
            print("Failed to perform move {}: {}; The random move {} will be performed instead ".format(randomMove, sys.exc_info(), randomMove))
            
    def evaluatePossibleMoves(self):        
        stockfishEvaluations = []
        simulatedTestData = pd.DataFrame(columns=["A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8",
                                          "A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7",
                                          "A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6",
                                          "A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5",
                                          "A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4",
                                          "A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3",
                                          "A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2",
                                          "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
                                          "Active Player",
                                          "White Kingside Castling Right", "White Queenside Castling Right", "Black Kingside Castling Right", "Black Queenside Castling Right",
                                          "En passant Target",
                                          "Halfmove Clock",
                                          "Fullmove Number",
                                          "Winning_Probability"])
        
        possibleMoves = self.chess.getPossibleMoves()
        result = pd.DataFrame(data=possibleMoves, columns=["Moves"])
        
        for move in possibleMoves:
            # Simulate a move
            self.chess.makeAMove(moveToString=move)
                
            simulatedTestData.loc[len(simulatedTestData)] = self.simulatedChessStateEncoder.createDataRecord()
            
            simulatedScore = self.simulatedChessStateEncoder.evaluateWithStockfish() * self.playerIndex
            stockfishEvaluations.append(simulatedScore)
                
            # Undo it, prepare for simulating the next move
            self.chess.unmakeAMove()
            
        # Store the evaluations of Stockfish
        stockfishEvaluations = pd.DataFrame(data=stockfishEvaluations, columns=["Stockfish_Evaluation"])
        result["Stockfish_Evaluation"] = stockfishEvaluations["Stockfish_Evaluation"]
            
        # Scale the simulations data
        simulatedTestData, simulatedTestDataMinColumnValues, simulatedTestDataMaxColumnValues = DataGenerator.minMaxScaling(data=simulatedTestData)
        
        # Predict the state evaluation of each simulation
        testScores = self.dnn.predict(X=simulatedTestData.iloc[:, :72].to_numpy().transpose())
        testScores = pd.DataFrame(testScores, columns=["Winning_Probability"])
        testScores["Winning_Probability"] *= self.playerIndex
        
        # Scale Stockfish evaluations data
        scaledStockfishEvaluations, _, _ = DataGenerator.minMaxScaling(data=stockfishEvaluations)
        
        # Sum of those scaled Data
        combinedEvaluations = pd.DataFrame(testScores["Winning_Probability"] + scaledStockfishEvaluations["Stockfish_Evaluation"], columns=["Combined_Evaluation"])
        
        # Descale simulations data
        originalTestScores = DataGenerator.minMaxDescaling(scaledData=testScores, minColumnValues=simulatedTestDataMinColumnValues, maxColumnValues=simulatedTestDataMaxColumnValues)
        
        result["Stockfish_Evaluation"] = stockfishEvaluations["Stockfish_Evaluation"]
        result["Winning_Probability"] = originalTestScores["Winning_Probability"]
        result["Scaled_Stockfish_Evaluation"] = scaledStockfishEvaluations["Stockfish_Evaluation"]
        result["Scaled_Winning_Probability"] = testScores["Winning_Probability"]
        result["Scaled_Combined_Evaluation"] = combinedEvaluations["Combined_Evaluation"]
        
        print("\n============================================================================")
        print("Possible moves and the corresponding Deep-Learning's evaluation")
        print(result)
            
        return result.loc[result["Scaled_Combined_Evaluation"].idxmax(), "Moves"]
            