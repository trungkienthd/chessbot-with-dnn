class Bot():

    def __init__(self, chess=None):
        self.chess = chess
        
        # Delay time to make the bot's more "smoothly"
        self.thinkingTime = 1.

    def __str__(self):
        raise NotImplementedError("This method should be overridden.")
        
    def perform(self):
        raise NotImplementedError("This method should be overridden.")
    
    @staticmethod
    def initializeBot(chess, botName="Random", playerIndex=1):
        if botName == "Random":
            from chessBots.randomBot import RandomBot
            
            return RandomBot(chess=chess)
        elif botName.startswith("DeepLearning_") and len(botName) > 13:
            from chessBots.deepLearningBot import DeepLearningBot
            
            deepLearningModelFileName = botName[13:]
            return DeepLearningBot(chess=chess, playerIndex=playerIndex, modelFileName=deepLearningModelFileName)
            
        else:
            print("\nBot {} is not defined. The Random bot is initialize instead.".format(botName))
            return RandomBot(chess=chess)