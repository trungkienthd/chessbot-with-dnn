import pyglet

from chessManager.point2d import Point2D
from chessManager.graphics import COLOUR_NAMES, window

PIECE_SPRITES = {
    " ": "default",
    
    "r": "blackRook",
    "n": "blackKnight",
    "b": "blackBishop",
    "q": "blackQueen",
    "k": "blackKing",
    "p": "blackPawn",
    
    "R": "whiteRook",
    "N": "whiteKnight",
    "B": "whiteBishop",
    "Q": "whiteQueen",
    "K": "whiteKing",
    "P": "whitePawn"
}

class Box:
    def __init__(self, xIndex=1, yIndex=1, boxSize=50., boardOffset=Point2D(50., 50.), firstColor="BLACK", secondColor="WHITE"):
        
        self.xIndex = xIndex
        self.yIndex = yIndex
        
        self.boxSize = boxSize
        
        self.boardOffset = boardOffset
        
        self.firstColor = firstColor
        self.secondColor = secondColor
        
        self.boxBackgroundSquare = pyglet.shapes.BorderedRectangle(
			x=self.worldPosition().x, y=self.worldPosition().y, width=self.boxSize, height=self.boxSize, border=1,
			color=COLOUR_NAMES[self.boxColor()], 
			border_color=COLOUR_NAMES["LIGHT_GREY"],
			batch=window.get_batch()
		)
        
        self.availableMoveIndication = pyglet.shapes.Circle(
            x=self.worldPosition().x + self.boxSize / 2, y=self.worldPosition().y + self.boxSize / 2,
            radius=5.,
            color=COLOUR_NAMES["TRANSPARENT"],
            batch=window.get_batch()
        )
        
        self.pieceCode = None
        self.boxPieceImage = None
        self.boxPieceSprite = None
        
        self.setBoxPieceSprite(pieceCode=" ")
        
    def boxColor(self):
        if self.xIndex % 2 == self.yIndex % 2:
            return self.firstColor
        else:
            return self.secondColor
        
    def worldPosition(self):
        return Point2D(self.boardOffset.x + self.boxSize * self.xIndex, self.boardOffset.y + self.boxSize * self.yIndex)
    
    def pointIsInBoxArea(self, x, y):
        if (x >= self.worldPosition().x) and (y >= self.worldPosition().y) and (x < self.worldPosition().x + self.boxSize) and (y < self.worldPosition().y + self.boxSize):
            return self
        
        return None
    
    def boardCoordinate(self):
        if 0 <= self.xIndex < 8 and 0 <= self.yIndex < 8:
            columnLetter = chr(self.xIndex + 97)  # Convert column index to letter (a-h)
            return "{}{}".format(columnLetter, self.yIndex + 1)
        else:
            print("Invalid indices")
            return ""
        
    def update(self, delta):
        return
    
    def setBoxPieceSprite(self, pieceCode):
        self.pieceCode = pieceCode
        self.boxPieceImage = pyglet.image.load("./chessManager/media/{}.png".format(PIECE_SPRITES[pieceCode]))
        self.boxPieceSprite = pyglet.sprite.Sprite(
            img=self.boxPieceImage,
            x=self.worldPosition().x, y=self.worldPosition().y,
            batch=window.get_batch()
        )
        
class Board:
    def __init__(self, chess=None, boardSize=400., boardOffset=Point2D(50., 50.), firstColor="BLACK", secondColor="WHITE"):
        self.chess = chess
        
        self.boxes = None
        self.chosenSourceBox = None
        self.chosenSourceBoxPossibleDestinations = []
        
        self.boxSize = boardSize / 8
        
        self.boardOffset = boardOffset
        
        self.firstColor = firstColor
        self.secondColor = secondColor
        
        self.initializeBoard()
        
    def initializeBoard(self):        
        self.boxes = []
        for xIndex in range(0, 8):
            self.boxes.append([])
            for yIndex in range(0, 8):
                self.boxes[xIndex].append(Box(xIndex=xIndex, yIndex=yIndex, boxSize=self.boxSize, boardOffset=self.boardOffset, firstColor=self.firstColor, secondColor=self.secondColor))
                
    def update(self, delta):
        for xIndex in range(0, len(self.boxes)):
            for yIndex in range(0, len(self.boxes[xIndex])):
                self.boxes[xIndex][yIndex].update(delta)

        # Reset all move's indications and chosen square
        for xIndex in range(0, len(self.boxes)):
            for yIndex in range(0, len(self.boxes[xIndex])):
                self.boxes[xIndex][yIndex].boxBackgroundSquare.border_color = COLOUR_NAMES[self.boxes[xIndex][yIndex].boxColor()]
                self.boxes[xIndex][yIndex].availableMoveIndication.color = COLOUR_NAMES["TRANSPARENT"]
                self.boxes[xIndex][yIndex].boxBackgroundSquare.border = 1
           
        self.chosenSourceBoxPossibleDestinations = []     
        if self.chosenSourceBox:
            # Set chosen square
            self.chosenSourceBox.boxBackgroundSquare.border_color = COLOUR_NAMES["RED"]
            self.chosenSourceBox.boxBackgroundSquare.border = 4
            
            # Set move's indications
            source = self.chosenSourceBox.boardCoordinate()
            movesFromSource = [move for move in self.chess.getPossibleMoves() if move[:2] == source]
            destinations = [move[2:4] for move in movesFromSource]
            
            # print("\n=======================================================================================")
            # print("Current possible moves: From {} to: {}".format(source, destinations))
            for xIndex in range(0, len(self.boxes)):
                for yIndex in range(0, len(self.boxes[xIndex])):
                    if self.boxes[xIndex][yIndex].boardCoordinate() in destinations:
                        self.chosenSourceBoxPossibleDestinations.append(self.boxes[xIndex][yIndex])
                        self.boxes[xIndex][yIndex].availableMoveIndication.color = COLOUR_NAMES["RED"]
            
    