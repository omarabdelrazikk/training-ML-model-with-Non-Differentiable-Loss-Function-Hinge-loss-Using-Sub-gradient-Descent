import numpy as np
class SVM_classifier():
    # initiating the hyperparameters
    def __init__(self,learning_rate,epochs,lambda_parameter,use_subgradient=True):
        self.use_subgradient = use_subgradient
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_parameter = lambda_parameter
        
    def fit(self,X,y,losspnt=False):
        
        # m  : number of data points 
        # n  : number of input features
        self.m , self.n = X.shape
        #initializing weights 
        self.w = np.random.rand(self.n)
        self.b = 0
        #recognizing inputs and class
        self.X = X
        self.y = y
        self.losses = []
        self.accuracies = []    
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        
        
        #  Optimization using Sub gradient Descent algorithm
        for i in range(self.epochs):
            self.update_weights()
            loss = self.calculate_hinge_loss(self.X, self.y)
            self.losses.append(loss)
            self.accuracies.append(self.calculate_accuracy(self.X,self.y))
            if (losspnt):
                print(f"Epoch {i+1}/{self.epochs} - Hinge Loss: {loss:.4f}, Accuracy: {self.accuracies[-1]:.2f}%, Precision: {self.calculate_precision(self.X,self.y):.2f}%, Recall: {self.calculate_recall(self.X,self.y):.2f}%, F1 Score: {self.calculate_f1_score(self.X,self.y):.2f}%")
                print("<-------------------------------------------------->")
            
    def update_weights(self):
        #label encoding {0:-1 , 1:+1}  
        y_label = np.where(self.y <= 0 , -1 ,1)
        
        # calculating gradients ( dw, db)
        for index , x_i in enumerate (self.X):
            decision_value = y_label[index] * (np.dot(x_i, self.w) + self.b)
            ##condition = y_label[index] * (np.dot(x_i, self.w) + self.b) >= 1 if self.use_subgradient else > 1
            if self.use_subgradient:
                condition = decision_value >= 1
            else:
                if decision_value == 1:
                    continue
                else :
                    condition = decision_value > 1
            # if the condition is satisfied, we update the weights and bias using the subgradient else we update the weights and bias using the gradient
            if condition :
                dw =  self.lambda_parameter * self.w
                db = 0
            else:
                dw = self.lambda_parameter * self.w - y_label[index] * x_i
                db = -y_label[index]
            # updating weights and bias
            self.w -= (self.learning_rate * dw)
            self.b -= (self.learning_rate * db)
     # calculating hinge loss to measure the performance of the SVM classifier   
    def calculate_hinge_loss(self, X, y):
        y_label = np.where(y <= 0, -1, 1)
        distances = 1 - y_label * (np.dot(X, self.w) + self.b)
        distances = np.maximum(0, distances)
        hinge_loss = np.mean(distances) + (self.lambda_parameter / 2) * np.dot(self.w, self.w)
        return hinge_loss 
    def calculate_accuracy(self,X,y):
        # calculating the accuracy of the model
        y_hat = self.predict(X)
        accuracy = np.mean(y_hat == y) * 100
        return accuracy
    def calculate_precision(self,X,y):
        # calculating the precision of the model
        y_hat = self.predict(X)
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_positive = np.sum((y_hat == 1) & (y == 0))
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        return precision * 100
    def calculate_recall(self,X,y):
        # calculating the recall of the model
        y_hat = self.predict(X)
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_negative = np.sum((y_hat == 0) & (y == 1))
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        return recall * 100
    def calculate_f1_score(self,X,y):
        # calculating the f1 score of the model
        precision = self.calculate_precision(X,y)
        recall = self.calculate_recall(X,y)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score
    
    def get_Weights(self):
        # returning the weights and bias
        return self.w , self.b
    def get_stats(self):
        # returning the losses for plotting the loss curve
        return self.losses ,self.accuracies , self.precisions , self.recalls , self.f1_scores
        
    
    # predict the label for a given input value
    def predict(self,X):
        output = np.dot(X,self.w) + self.b
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= -1 ,0 ,1)
        return y_hat