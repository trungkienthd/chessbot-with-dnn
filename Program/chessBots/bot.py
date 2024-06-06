class Bot():

    def __init__(self, chess=None):
        self.chess = chess
        
        # Delay time to make the bot's more "smoothly"
        self.thinkingTime = 1.

    def __str__(self):
        raise NotImplementedError("This method should be overridden.")
        
    def perform(self):
        raise NotImplementedError("This method should be overridden.")