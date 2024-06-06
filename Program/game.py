import sys
import pyglet

from chessManager.graphics import window
from chessManager.point2d import Point2D
from chessManager.board import Board
from chessManager.chess import Chess
from chessBots.randomBot import RandomBot

class Game():
    def __init__(self):
        self.chess = Chess()
        self.board = Board(chess=self.chess, boardSize=640., boardOffset=Point2D(80., 80.), firstColor="BLUE", secondColor="WHITE")
        
        self.bot = RandomBot(chess=self.chess)
        
        self.turn = "Player"
        self.botThinkingTimer = 0.

    def inputMouse(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT and self.turn == "Player":
            clickedBox = None
            for xIndex in range(0, len(self.board.boxes)):
                for yIndex in range(0, len(self.board.boxes[xIndex])):
                    clickedBox = self.board.boxes[xIndex][yIndex].pointIsInBoxArea(x, y)
                    if clickedBox:
                        break
                if clickedBox:
                    break
                    
            if clickedBox:
                # Deselect the clickedBox
                if self.board.chosenSourceBox == clickedBox:
                    self.board.chosenSourceBox = None
                # Choose a box
                elif clickedBox.pieceCode != " " and not clickedBox in self.board.chosenSourceBoxPossibleDestinations:
                    self.board.chosenSourceBox = clickedBox
                # Choose a destination for the chosen box
                elif clickedBox in self.board.chosenSourceBoxPossibleDestinations:
                    # print("{}{}".format(self.board.chosenSourceBox.boardCoordinate(), clickedBox.boardCoordinate()))
                    self.chess.makeAMove(moveToString="{}{}".format(self.board.chosenSourceBox.boardCoordinate(), clickedBox.boardCoordinate()))
                    self.board.chosenSourceBox = None
                    
                    self.turn = "Bot"

    def inputKeyboard(self, symbol, modifiers):
        return

    def update(self, delta):      
        try:
            boardToCode = self.chess.extractFEN()
            for xIndex in range(0, len(boardToCode)):
                for yIndex in range(0, len(boardToCode[xIndex])):
                    self.board.boxes[xIndex][yIndex].setBoxPieceSprite(pieceCode=boardToCode[xIndex][yIndex])
        except:
            print("FAILED TO CONVERT BOARD'S CODES TO IMAGES: {}".format(sys.exc_info()[0]))
                
        self.board.update(delta)
        window.updateLabel("bot's thinking timer", "ETA Thinking time: {} secs.".format((self.bot.thinkingTime - self.botThinkingTimer), ".2f"))
        
        if self.turn == "Bot":
            self.botThinkingTimer += delta
            if self.botThinkingTimer >= self.bot.thinkingTime:
                self.bot.perform()
                self.turn = "Player"
        elif self.turn == "Player":
            self.botThinkingTimer = 0
    
game = None