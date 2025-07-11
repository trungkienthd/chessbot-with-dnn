import sys
import random
import chess

class Chess():
    
    def __init__(self):
        self.board = None
        self.initializeChessBoard()
        
    def initializeChessBoard(self):
        self.board = chess.Board()
        
    def getFen(self):
        return self.board.fen()
    
    def getPossibleMoves(self):  
        return [move.uci() for move in self.board.legal_moves]
    
    def makeAMove(self, moveToString):
        if moveToString in self.getPossibleMoves():
            try:
                self.board.push_uci(moveToString)
                print("\n=======================================================================================")
                print("Move {} performed successfully.".format(moveToString))
                print("Current board's FEN: {}".format(self.board.fen()))
            except:
                print("Failed to perform move {}: {}".format(moveToString, sys.exc_info()))
        else:
            print("Invalid move: " + moveCode)
            
        return
    
    def unmakeAMove(self):
        try:
            self.board.pop()
            print("\n=======================================================================================")
            print("Last move has been unmade successfully.")
        except:
            print("Failed to unmake the last move: {}".format(sys.exc_info()))

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