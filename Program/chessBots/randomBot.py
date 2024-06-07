import sys
import random

from chessBots.bot import Bot

class RandomBot(Bot):
    
    def __init__(self, chess=None):
        super().__init__(chess=chess)
        
        self.thinkingTime = 1.
        
    def __str__(self):
        return "RandomBot"
    
    def perform(self):
        randomMove = random.choice(self.chess.getPossibleMoves())
        try:
            self.chess.makeAMove(moveToString=randomMove)
        except:
            print("Failed to perform random move {}: {} ".format(randomMove, sys.exc_info()))
            