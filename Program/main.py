import pyglet

import game
from deepLearningAI.training.dataProcessing import DataGenerator

if __name__ == '__main__':    
    data = DataGenerator(whiteBot="Random", blackBot="Random", numberOfSimulations=10)
    game.game = game.Game()
    pyglet.clock.schedule_interval(game.game.update, 1/60.)
    pyglet.app.run()
