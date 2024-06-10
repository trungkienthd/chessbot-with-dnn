import numpy as np

# ===============================================================================================================================
# LINEAR LAYER CLASS
class Linear():
    
    def __init__(self, inputDimension, outputDimension, regularization=0.0, learningRate=3e-6):
        
        # inputDimension means the number of units in the previous hidden layer, or the input size of the input layer (if that is the previous layer)
        # outputDimension also means the number of units in the current layer
        self.numberOfUnits = outputDimension
        
        self.W = np.random.rand(outputDimension, inputDimension) - 0.5
        self.b = np.random.rand(outputDimension, 1) - 0.5
        
        self.cache = None
        
        self.regularization = regularization
        self.learingRate = learningRate
        
    def forward(self, A):
        
        # A is the output from the previous hidden layer with the shape of (<number of units in that previous layer>, <number of examples>)
        # A means X (matrix of examples) if the previous layer is the input layer
        
        # Cache is stored for computing the backward pass efficiently
        self.cache = (A, self.W, self.b)
        
        # Z is the result matrix after the linear transformation process, with the shape of (<number of units in the current layer>, <number of units in the previous hidden layer>)
        # The brief formula is Z = A * W + b
        
        # Calculate simply using numpy matrix multiplication:
        Z = np.dot(self.W, A) + self.b
        
        # print("=========================Z - SIMPLE CALCULATING=========================")
        # print(Z)
        
        # Calculate properly:
        # (previousLayerDimension, numberOfExamples) = A.shape
        
        # currentLayerDimentsion = self.W.shape[0]
        
        # Z = np.zeros((currentLayerDimentsion, numberOfExamples))
        
        # for i in range(currentLayerDimentsion):
        #     for j in range(numberOfExamples):
        #         dot = 0
        #         for k in range(previousLayerDimension):
        #             dot += A[k, j] * self.W[i, k]
                    
        #         # Add the bias
        #         Z[i, j] = dot + self.b[i]
                
        # Example:
        #   Number of examples: m = 4
        #   Current layer's number of units: n^[l] = 3
        #   Previous layer's number of units: n^[l-1] = 2
        #
        #   A with shape (3, 4)
        #       [ [ x_00, x_01, x_02, x_03 ],
        #         [ x_10, x_11, x_12, x_13 ],
        #         [ x_20, x_21, x_22, x_23 ] ]   
        # 
        #   W with shape (2, 3)
        #   [ [ W_00, W_01, W_02 ],
        #     [ W_10, W_11, W_12 ] ]
        # 
        #   b with shape (2, 1)
        #       [ [ b_0 ], 
        #         [ b_1] ]
        #
        #   => Z with shape (2, 4)
        #       [ [ (W_00 * x_00 + W_01 * x_10 + W_02 * + x_20 + b_0), (W_00 * x_01 + W_01 * x_11 + W_02 * + x_21 + b_0), (W_00 * x_02 + W_01 * x_12 + W_02 * + x_22 + b_0), (W_00 * x_03 + W_01 * x_13 + W_02 * + x_23 + b_0) ],
        #         [ (W_10 * x_00 + W_11 * x_10 + W_12 * + x_20 + b_1), (W_10 * x_01 + W_11 * x_11 + W_12 * + x_21 + b_1), (W_10 * x_02 + W_11 * x_12 + W_12 * + x_22 + b_1), (W_10 * x_03 + W_11 * x_13 + W_12 * + x_23 + b_1) ] ]
                
        # print("=======================Z - PROPERLY CALCULATING=========================")                 
        # print(Z)
                
        return Z
    
    def backward(self, deltaZ):
        
        # deltaZ is the gradient of the cost with respect to the output of the current linear layer
        # Alternatively, given that L is the cost function, Z is the outputs of the current linear layer
        #   deltaZ = dL / dZ
        
        # Get the previous input
        A = self.cache[0]
         
        # Get the number of examples 
        # m = A.shape[1]
        
        # Calculate the gradients:
        #   deltaW is the gradient of the cost with respect to the weights W of the current linear layer (dZ / dW)
        #       Assume that f is the cost function based on the output (L = f(Z)), g is the function to calculate the output based on W (Z = g(W))
        #       (The "chain rule")
        #       From L = f(Z) = f(g(W)), I can state that: [f(g(x))]' = f'(g(x)) * g'(x)  <=>  dL / dW = (dL / dZ) * (dZ / dW)  <=>  deltaW = deltaZ * (dZ / dW)
        #       Also, in the forward process I have implemented: Z = A * W + b. That is why: dZ / dW = d(A * W + b) / dW = A
        #       Conclude: deltaW = deltaZ * A

        deltaW = np.dot(deltaZ, A.transpose()) + self.regularization * self.W
        
        #   deltab is the gradient of the cost with respect to the biases b of the current linear layer (dZ / db)
        #       With a similar explanation presented above, I can state that dL / db = (dL / dZ) * (dZ / db) <=> deltab = deltaZ * (dZ / db)
        #       In the forward process I have implemented: Z = A * W + b. That is why: dZ / db = d(A * W + b) / db = 1
        #       Conclude: deltab = deltaZ
        
        deltab = (np.sum(deltaZ, axis=1))
        deltab = deltab.reshape((deltab.shape[0], 1))
        deltab += (self.regularization * self.b)
        
        #   dA is the gradient of the cost with respect to the input of the current linear layer (which is also the output of the previous layer)
        #       This is similar to what has happened when I caculating deltaW = dL / dA = (dL / dZ) * (dZ / dA) = deltaZ * [d(A * W + b) / dA] = deltaZ * W
        deltaA = np.dot(self.W.transpose(), deltaZ)
        
        # It can be easily duduced that deltaZ, deltaW, deltab, and deltaA have the same shape as Z, W, b, and A respectively.
        # So we simply use the numpy.transpose effectively before multiplicating the matrices to make sure that all of those required output have the precise shape. 
        # Take an example described in the forward process above: 
        #   Number of examples: m = 4
        #   Current layer's number of units: n^[l] = 3
        #   Previous layer's number of units: n^[l-1] = 2
        #   => deltaZ.shape = (2, 4), deltaW.shape = (2, 3), deltab.shape = (3, 1)
        
        # print("Back deltaZ: {}".format(deltaZ.shape))
        # #print("Back deltaZ: {}".format(deltaZ))
        # print("Back deltaW: {}".format(deltaW.shape))
        # print("Back deltab: {}".format(deltab.shape))
        # print("Back deltaA: {}\n".format(deltaA.shape))
        #print("Back deltaA: {}\n".format(deltaA))
        
        # print("Back W before updating: {}".format(self.W))
        
        # Update the weights and biases (Gradient descent)
        self.W -= self.learingRate * deltaW
        self.b -= self.learingRate * deltab
        
        # print("Back W after updating: {}".format(self.W))
        
        return (deltaA, deltaW, deltab)
    
# ===============================================================================================================================
# ReLU ACTIVATION FUNCTION CLASS
class ReLU():
    
    def __init__(self):
        self.cache = None
        
    def __str__(self):
        return "ReLU"
    
    def forward(self, A):

        # Cache is stored for computing the backward pass efficiently
        self.cache = A
        
        # ReLU activation function is stated: f(x) = max(0, x)
        
        # Calculate the output of this ReLU layer
        Z = np.maximum(0, A)
        
        # print("===================================Z===================================")
        # print(Z)
        
        return Z
    
    def backward(self, deltaZ):
        
        # deltaZ is the gradient of the cost with respect to the output of this ReLU layer (dL / dZ)
        
        # Get the previous input
        A = self.cache
        
        # Given that g(x) is ReLU activation function, g'(x) means:
        #   If x < 0, g'(x) = 0
        #   If x > 0, g'(x) = 1
        #   g'(x) is not defined at x = 0
        
        # deltaA is the gradient of the cost with respect to the input of this ReLU layer (dL / dA)
        # Using the chain rule, it is easy to see that: dL / dA = (dL / dZ) * (dZ / dA)
        #   Initially, calculate the gradient of the output with respect to this ReLU layer's input (dZ / dA) based on the g'(x) explanation above:
        gradientZToA = (A > 0).astype(np.float32)
        
        # print("==========================GRADIENT OF Z TO A============================")     
        # print(gradientZToA)
        
        #   Then, calculate deltaA:
        deltaA = deltaZ * gradientZToA
        
        # print("===============================DELTA-A==================================")     
        # print(deltaA)
        
        return deltaA     
    
# ===============================================================================================================================
# SIGMOID ACTIVATION FUNCTION CLASS
class Sigmoid():
    
    def __init__(self):
        self.cache = None
        
    def __str__(self):
        return "Sigmoid"
        
    def forward(self, A):
        
        # Cache is stored for computing the backward pass efficiently
        self.cache = A
        
        Z = 1 / (1 + np.exp(-A))
        
        return np.array(Z)
    
    def backward(self, deltaZ):
        
        # Get the previous input
        A = self.cache
        
        # Calculate the gradient of the output with the respect to this Sigmoid layer's input (dZ / dA):
        gradientZToA = np.exp(-A) / ((1 + np.exp(-A)) ** 2)
        
        # deltaA is the gradient of the cost with respect to the input of this Sigmoid layer (dL / dA)
        # Using the chain rule, it is easy to see that: dL / dA = (dL / dZ) * (dZ / dA) = deltaZ * gradientZToA
        deltaA = deltaZ * gradientZToA
        
        return deltaA

# ===============================================================================================================================
# GAUSSIAN ACTIVATION FUNCTION CLASS
class Gaussian():
    
    def __init__(self):
        self.cache = None
        
    def __str__(self):
        return "Gaussian"
    
    def forward(self, A):

        # Cache is stored for computing the backward pass efficiently
        self.cache = A
        
        # ReLU activation function is stated: f(x) = e^(-x^2)
        
        # Calculate the output of this ReLU layer
        Z = np.exp(-A**2)
        
        # print("===================================Z===================================")
        # print(Z)
        
        return Z
    
    def backward(self, deltaZ):
        
        # Get the previous input
        A = self.cache

        # Calculate the gradient of the output with the respect to this Gaussian layer's input (dZ / dA):
        gradientZToA = -2 * A * np.exp(-A**2)
        
        # deltaA is the gradient of the cost with respect to the input of this Gaussian layer (dL / dA)
        # Using the chain rule, it is easy to see that: dL / dA = (dL / dZ) * (dZ / dA) = deltaZ * gradientZToA
        deltaA = deltaZ * gradientZToA
        
        return deltaA
  
# ===============================================================================================================================
# CLASS FOR LAYER THAT COMBINES LINEAR LAYER AND ACTIVATION-FUNCTION LAYER
class StandardLayer():
    
    def __init__(self, inputDimension, outputDimension, learningRate, regularization, activationFunction):
        
        self.linearLayer = Linear(inputDimension=inputDimension, outputDimension=outputDimension, learningRate=learningRate, regularization=regularization)
        
        self.activationFunctionLayer = None
        if (activationFunction == "ReLU"):
            self.activationFunctionLayer = ReLU()
        elif (activationFunction == "Sigmoid"):
            self.activationFunctionLayer = Sigmoid()
        elif (activationFunction == "Gaussian"):
            self.activationFunctionLayer = Gaussian()
        else:
            print("Activation Function {} is not defined. ReLU function is registered instead.".format(activationFunction))
            self.activationFunctionLayer = ReLU()    
            
    def forward(self, A):
        
        hidden = self.linearLayer.forward(A)
        Z = self.activationFunctionLayer.forward(hidden)
        
        return Z      
    
    def backward(self, deltaZ):
        
        # deltaZ is the gradient of the cost with the respect to the post-activation output
        # deltaHidden is the gradient of the cost with the respect to the post-linear function output
        
        deltaHidden = self.activationFunctionLayer.backward(deltaZ)
        (deltaA, deltaW, deltab) = self.linearLayer.backward(deltaZ)
        
        return (deltaA, deltaW, deltab)
   
# ===============================================================================================================================
# CLASS TO COMPUTE THE COST USING BINARY CROSS ENTROPY 
class BinaryCrossEntropy():
    
    def __init__(self):
        self.cache = None
    
    def forward(self, scores, labels):
        
        self.cache = scores
        
        # Scores are the prediction values, while labels are the corresponding actual values
        numberOfExamples = scores.shape[0]
        
        # Binary Cross Entropy formula:
        # J = (-1 / m) * [(y_1 * log(a_1) + (1 - y_1) * log (1 - a_1)) + (y_2 * log(a_2) + (1 - y_2) * log (1 - a_2)) + ... + (y_m * log(a_m) + (1 - y_m) * log (1 - a_m))]
        cost = (np.dot(labels, np.log(scores).transpose()) + np.dot((1 - labels), np.log(1 - scores).transpose()))

        cost *= (-1. / numberOfExamples)
        
        return cost[0]
    
    def backward(self, labels):
        
        # Based on the Binary Cross Entropy formula, foreach pair of (a_i, y_i), the gradient of the cost with respect to a_i is J' can be computed as following:                             
        # J' =  (-1 / m) * [(const)' + ((y_i * log(a_i) + (1 - y_i) * log (1 - a_i)))' ]
        #    =  (-1 / m) * [0 + (y / a) + (-(1 - y) / (1 - a))]     
        #    =  (-1 / m) * (y - a) / [a * (1 - a)]
        
        scores = self.cache
        
        numberOfExamples = scores.shape[0]
        
        # deltaAL is the gradient of the cost with respect to scores
        costGradient = 0.
        for i in range(0, min(len(scores), len(labels))):
            costGradient += ((labels[i] - scores[i]) / (scores[i] * (1 - scores[i])))
            
        costGradient *= (-1. / numberOfExamples)
        
        return np.array([costGradient])
            
        
    
# ===============================================================================================================================
# LINEAR TEST
# linear = Linear(3, 2)

# A = np.random.rand(3, 4) * 0.01

# print("===================================A====================================") 
# print(A)
# print("===================================W====================================") 
# print(linear.W)

# linear.forward(A)

# ===============================================================================================================================
# ReLU TEST
# reLU = ReLU()

# A = np.random.rand(3, 4)
# deltaZ = np.random.randn(3, 4)

# print("===================================A====================================") 
# print(A)
# print("================================DELTA-Z=================================")
# print(deltaZ)

# reLU.forward(A)
# reLU.backward(deltaZ)
