import sys
import pyglet

import game
from deepLearningAI.training.dataGenerating import DataGenerator
from deepLearningAI.training.training import Trainer

def play(modelFileName="ModelDataFile-'1_Simulations_Of_White_RandomBot_VS_Black_RandomBot'__HiddenLayersActivationFunction-'ReLU'__Epoches-'150'__MiniBatchGD-'64'.pickle"):
    game.game = game.Game(modelFileName=modelFileName)
    pyglet.clock.schedule_interval(game.game.update, 1/60.)
    pyglet.app.run()
    
if __name__ == '__main__':  
    print("================================================================================================================================================")
    print("================================================================================================================================================")
    print("================================================================================================================================================")
    print("================================================================================================================================================")
    print("================================================================================================================================================")
    print()
    if len(sys.argv) >= 2:  
        if len(sys.argv) == 5 and sys.argv[1] == "generateData":
            # E.g. "python main.py generateData DeepLearning_ModelDataFile-'1_Simulations_Of_White_RandomBot_VS_Black_RandomBot'__HiddenLayersActivationFunction-'ReLU'__Epoches-'150'__MiniBatchGD-'64'.pickle Random 1"
            try:
                numberOfSimulations = (int)(sys.argv[4])
                data = DataGenerator(whiteBot=sys.argv[2], blackBot=sys.argv[3], numberOfSimulations=numberOfSimulations)
            except:
                print("Unable to generate data: {}. The game will start instead".format(sys.exc_info()))
                play()
        elif len(sys.argv) == 6 and sys.argv[1] == "train":
            # E.g. "python main.py train 1_Simulations_Of_White_RandomBot_VS_Black_RandomBot ReLU 1"
            try:
                dataFileName = sys.argv[2]
                activationFunctionsOfHiddenLayers = sys.argv[3]
                numberOfEpoches = (int)(sys.argv[4])
                batchSize = (int)(sys.argv[5])
                
                trainer = Trainer(dataFileName=dataFileName, activationFunctionsOfHiddenLayers=activationFunctionsOfHiddenLayers, numberOfEpoches=numberOfEpoches, batchSize=batchSize)
            except:
                print("Unable to train Neural network: {}. The game will start instead".format(sys.exc_info()))
                play()
        elif len(sys.argv) == 2:    
            try:
                modelFileName = sys.argv[1]
                play(modelFileName=modelFileName)
            except:
                print("Unable to start the game with the model {}: {}. The game will start with the default bot instead".format(modelFileName, sys.exc_info()))
                play()
        else:
            print("System arguments are not defined. The game will start instead")
            play()
    else:
        play()
