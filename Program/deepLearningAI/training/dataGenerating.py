import numpy as np
import pandas as pd

import chess
import chess.engine

from chessManager.chess import Chess

from chessBots.bot import Bot

PIECE_TO_VALUE = {
        chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
        -chess.PAWN: -1, -chess.KNIGHT: -2, -chess.BISHOP: -3, -chess.ROOK: -4, -chess.QUEEN: -5, -chess.KING: -6
    }

class ChessStateEncoder():
    
    def __init__(self, board):
        self.board = board
        
        self.encodedFenArray = None
        # self.initializeEncodedFenArray()
        
    def initializeEncodedFenArray(self):
        # BOARD STATE FEATURES
        boardStateMatrix = np.zeros((8, 8), dtype=int)
        # Encoding the pieces on the board
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                color_multiplier = 1 if piece.color == chess.WHITE else -1
                boardStateMatrix[i // 8, i % 8] = color_multiplier * PIECE_TO_VALUE[piece.piece_type]
                
        boardStateVector = []
        for file in boardStateMatrix:
            for element in file:
                boardStateVector.append(element)
                
        # ADDITIONAL FEATURES
        
        # Active player (w - white or b - black): Encoded as 1 for white, -1 for black
        encodedActivePlayer = 1 if self.board.turn == chess.WHITE else -1
        
        # Castiling Availability: Encoded into four pairs of positions representing K, Q, k, q. Each available castling right is marked as pair of 1 values, otherwise pair of 0 values
        encodedWhiteKingsideCastlingRights = 1 if self.board.has_kingside_castling_rights(chess.WHITE) else 0
        encodedWhiteQueensideCastlingRights = 1 if self.board.has_queenside_castling_rights(chess.WHITE) else 0
        encodedBlackKingsideCastlingRights = 1 if self.board.has_kingside_castling_rights(chess.BLACK) else 0
        encodedBlackQueensideCastlingRights = 1 if self.board.has_queenside_castling_rights(chess.BLACK) else 0
        
        # En passant Target: The potential square where an en passant capture can occur
        encodedEnPassantTarget = -1
        if self.board.ep_square:
            encodedEnPassantTarget = self.board.ep_square
            
        # Halfmove clock: This is a number indicating the number of halfmoves (or ply) since the last capture or pawn advance. This is used to determine if a draw can be claimed under the fifty-move rule.
        encodedHalfmoveClock = self.board.halfmove_clock
        
        # Fullmove number: This represents the total number of full moves in the game, starting at 1, and increasing after Black's move.
        encodedFullmoveNumber = self.board.fullmove_number
        
        self.encodedFenArray = (boardStateVector
                                + [encodedActivePlayer]
                                + [encodedWhiteKingsideCastlingRights, encodedWhiteQueensideCastlingRights, encodedBlackKingsideCastlingRights, encodedBlackQueensideCastlingRights]
                                + [encodedEnPassantTarget]
                                + [encodedHalfmoveClock]
                                + [encodedFullmoveNumber])
        
        # print(self.encodedFenArray)
        self.encodedFenArray = np.array(self.encodedFenArray)
        
        # self.printEncodedFEN(boardStateMatrix=boardStateMatrix)
        
        return self.encodedFenArray
    
    # A positive score <=> “White is likely winning” 
    # A negative socre <=> “Black is likely winning”.
    def evaluateWithStockfish(self):
        score = 0
        with chess.engine.SimpleEngine.popen_uci('./deepLearningAI/training/stockfish/stockfish-windows-x86-64-avx2.exe') as sf:
            result = sf.analyse(self.board, chess.engine.Limit(time=0.1))
            score = result['score'].white().score()
            
            # Handling None value
            if score is None:
                score = 0
                
        return score
        
    # Create Record of Encoded board's state and its Evaluation    
    def createDataRecord(self):
        score = self.evaluateWithStockfish()
            
        record =  np.concatenate((self.initializeEncodedFenArray(), np.array([score])))     
        
        # print(" => Score: {}\n".format(score))   
        
        return record
        
    def printEncodedFEN(self, boardStateMatrix):
        print("\n=======================================================================================")
        print("FEN '{}' has been encoded:".format(self.board.fen()))
        print("     ------------------------------------------------------------------------")
        for i in range(0, len(boardStateMatrix)):
            print("     Board state - File {}:   {}".format(chr(i + 65), ["{:+.0f}".format(number) for number in boardStateMatrix[i]]))
        print("     ------------------------------------------------------------------------")
        print("     Active player (w/b):    {}".format(self.encodedFenArray[-6]))
        print("     ------------------------------------------------------------------------")
        print("     Castiling availability: {}".format(self.encodedFenArray[-7:-3]))
        print("     ------------------------------------------------------------------------")
        print("     En passant Target:      {}".format(self.encodedFenArray[-3]))
        print("     ------------------------------------------------------------------------")
        print("     Halfmove Clock:         {}".format(self.encodedFenArray[-2]))
        print("     ------------------------------------------------------------------------")
        print("     Fullmove number:        {}".format(self.encodedFenArray[-1]))
        print("     ------------------------------------------------------------------------")
        
        print(" => Encoded FEN Array: {} ({} elements)".format(self.encodedFenArray, len(self.encodedFenArray)))

class DataGenerator():
    
    def __init__(self, whiteBot="Random", blackBot="Random", numberOfSimulations=1):
        
        self.chess = Chess()
        
        self.whiteRandomBot = Bot.initializeBot(botName=whiteBot, playerIndex=1)
        self.blackRandomBot = Bot.initializeBot(botName=blackBot, playerIndex=-1)
        
        self.turn = "White"
        
        self.chessStateEncoder = ChessStateEncoder(self.chess.board)
        
        self.data = pd.DataFrame(columns=["A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8",
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
                                          "Who is winning"])
        
        for i in range(0, numberOfSimulations):
            self.simulate()
           
        self.data, _, _ = DataGenerator.minMaxScaling(data=self.data)  
        self.data.to_csv("./deepLearningAI/data/{}_Simulations_Of_White_{}_VS_Black_{}.csv".format(numberOfSimulations, self.whiteRandomBot, self.blackRandomBot), index=False)
        self.printData()
        
    def simulate(self):
        self.chess = Chess()
        self.whiteRandomBot.chess = self.chess
        self.blackRandomBot.chess = self.chess
        self.chessStateEncoder = ChessStateEncoder(self.chess.board)
        
        while not self.chess.board.is_game_over():
            if self.turn == "White":
                self.whiteRandomBot.perform()
                self.turn = "Black"
            elif self.turn == "Black":
                self.blackRandomBot.perform()
                self.turn = "White"
            
            self.data.loc[len(self.data)] = self.chessStateEncoder.createDataRecord()
                    
    def printData(self):
        print("\n=======================================================================================")
        print("Data generated when simulating a match between {} (White) and {} (Black), after applying Min-Max Scaling:".format(self.whiteRandomBot, self.blackRandomBot))
        print(self.data)
    
    @staticmethod
    def minMaxScaling(data):
        # Copy the dataframe to avoid changing the original data
        scaledData = data.copy()
        
        minColumnValues = {}
        maxColumnValues = {}
        
        # Min-Max Scaling (Normalization) rescales the data to a fixed range, typically 0 to 1, or -1 to 1; using the following formula:
        # X_norm = (X - X_min) / (X_max - X_min)
        
        # Apply Min-Max Scaling to each column
        for column in data.columns:
            minColumnValues[column] = data[column].min()
            maxColumnValues[column] = data[column].max()
            
            minmaxRangeValue = maxColumnValues[column] - minColumnValues[column]
            # Avoid division by 0 by using np.where to replace zero values in minmaxRangeValue with 1 
            minmaxRangeValue = np.where(minmaxRangeValue == 0, 1, minmaxRangeValue)
            
            scaledData[column] = (data[column] - minColumnValues[column]) / (minmaxRangeValue)
            
        return scaledData, minColumnValues, maxColumnValues
    
    @staticmethod
    def minMaxDescaling(scaledData, minColumnValues, maxColumnValues):
        # minColumnValues (dict): Dictionary containing the minimum values for each column used during scaling.
        # maxColumnValues (dict): Dictionary containing the maximum values for each column used during scaling.
        
        originalData = scaledData.copy()
        for column in scaledData.columns:
            minmaxRangeValue = maxColumnValues[column] - minColumnValues[column]
            minmaxRangeValue = np.where(minmaxRangeValue == 0, 1, minmaxRangeValue)
            
            originalData[column] = scaledData[column] * minmaxRangeValue + minColumnValues[column]

        return originalData
