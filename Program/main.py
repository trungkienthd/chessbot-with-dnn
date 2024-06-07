import sys
import pyglet

import game
from deepLearningAI.training.dataProcessing import DataGenerator

def play():
    game.game = game.Game()
    pyglet.clock.schedule_interval(game.game.update, 1/60.)
    pyglet.app.run()
    
if __name__ == '__main__':  
    if len(sys.argv) >= 2:  
        if len(sys.argv) == 5 and sys.argv[1] == "generateData":
            try:
                numberOfSimulations = int.Parse(sys.argv[4])
                data = DataGenerator(whiteBot=sys.argv[2], blackBot=sys.argv[3], numberOfSimulations=1)
            except:
                print("Unable to generate data: {}".format(sys.exc_info()[0]))
                play()
        elif len(sys.argv) == 3 and sys.argv[1] == "train":
            return
        else:
            print("System arguments are not defined! The game will start instead")
            play()
    else:
        play()
