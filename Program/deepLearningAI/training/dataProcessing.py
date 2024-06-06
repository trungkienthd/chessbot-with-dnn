import numpy as np
import pandas as pd

import chess

from chessManager.chess import Chess
from chessBots.randomBot import RandomBot

PIECE_TO_VALUE = {
        chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
        -chess.PAWN: -1, -chess.KNIGHT: -2, -chess.BISHOP: -3, -chess.ROOK: -4, -chess.QUEEN: -5, -chess.KING: -6
    }

class BoardEncoder():
    
    def __init__(self, board):
        self.board = board
        
        self.encodedFenArray = None
        self.initializeEncodedFenArray()
        
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
        
        print(self.encodedFenArray)
        self.encodedFenArray = np.array(self.encodedFenArray)
        
        self.printEncodedFEN(boardStateMatrix=boardStateMatrix)
        
        return self.encodedFenArray
        
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
        
        self.whiteRandomBot = RandomBot(chess=self.chess)
        self.blackRandomBot = RandomBot(chess=self.chess)
        
        self.turn = "White"
        
        self.boardEncoder = BoardEncoder(self.chess.board)
        
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
                                          "Fullmove Number"])
        
        for i in range(0, numberOfSimulations):
            self.simulate()
            
        self.data.to_csv("./deepLearningAI/data/{}_Simulations_Of_White_{}_VS_Black_{}.csv".format(numberOfSimulations, self.whiteRandomBot, self.blackRandomBot), index=False)
        self.printData()
        
    def simulate(self):
        self.chess = Chess()
        self.whiteRandomBot.chess = self.chess
        self.blackRandomBot.chess = self.chess
        self.boardEncoder = BoardEncoder(self.chess.board)
        
        while not self.chess.isGameOver():
            if self.turn == "White":
                self.whiteRandomBot.perform()
                self.turn = "Black"
            elif self.turn == "Black":
                self.blackRandomBot.perform()
                self.turn = "White"
                
            self.data.loc[len(self.data)] = self.boardEncoder.initializeEncodedFenArray()
            
    def printData(self):
        print("\n=======================================================================================")
        print("Data generated when simulating a match between {} (White) and {} (Black):".format(self.whiteRandomBot, self.blackRandomBot))
        print(self.data)
