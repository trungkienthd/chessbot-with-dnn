import sys
import random
import chess

class Chess():
    
    def __init__(self):
        self.board = None
        self.initializeChessGame()
        
    def initializeChessGame(self):
        self.board = chess.Board()
        
    def getFen(self):
        return self.board.fen()
    
    def getPossibleMoves(self):  
        return [move.uci() for move in self.board.legal_moves]
    
    def isGameOver(self):
        return (self.board.is_checkmate() or
                self.board.is_stalemate() or
                self.board.is_insufficient_material() or
                self.board.is_seventyfive_moves() or  # Optionally, checking the 75-move rule
                self.board.is_fivefold_repetition() or  # Optionally, checking for fivefold repetition
                self.board.can_claim_draw())  # General draw claim (includes threefold repetition and fifty-move rule)
    
    def makeAMove(self, moveToString):
        if moveToString in self.getPossibleMoves():
            try:
                self.board.push_uci(moveToString)
                print("Move {} performed successfully.".format(moveToString))
                # print("\n=======================================================================================")
                # print("Current board's FEN: {}".format(self.board.fen()))
            except:
                print("Failed to perform move {}: {}".format(moveToString, sys.exc_info()[0]))
        else:
            print("Invalid move: " + moveCode)
            
        return

    def extractFEN(self):
        rows = self.board.fen().split(" ")[0].split("/")
        boardToString = []
        for row in rows:
            rowSquares = []
            for char in row:
                if char.isdigit():
                    # Add multiple empty strings for consecutive empty squares
                    rowSquares.extend([" "] * int(char))
                else:
                    # Add the piece character
                    rowSquares.append(char)
            boardToString.append(rowSquares)
            
        boardToString = boardToString[::-1]
            
        # print("\n=======================================================================================")
        # print("Current board state: ")
        # for row in boardToString:
        #     print(row)
        
        return [list(col) for col in zip(*boardToString)]