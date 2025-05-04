import numpy as np

class SVM_classifier():
    def __init__(self, learning_rate, epochs, lambda_parameter, use_subgradient=True,
                 kernel_type='linear', gamma=0.1, degree=3):
        self.use_subgradient = use_subgradient
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_parameter = lambda_parameter
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree

    def kernel(self, x1, x2):
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel_type == 'poly':
            return (np.dot(x1, x2) + 1) ** self.degree
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, X, y, losspnt=False):
        self.m, self.n = X.shape
        self.alpha = np.zeros(self.m)
        self.b = 0
        self.X = X
        self.y = y
        self.losses = []
        self.accuracies = []    
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

        for i in range(self.epochs):
            self.update_weights()
            loss = self.calculate_hinge_loss(self.X, self.y)
            self.losses.append(loss)
            self.accuracies.append(self.calculate_accuracy(self.X, self.y))
            self.precisions.append(self.calculate_precision(self.X, self.y))
            self.recalls.append(self.calculate_recall(self.X, self.y))
            self.f1_scores.append(self.calculate_f1_score(self.X, self.y))

            if losspnt and i % 1000 == 0:
                print(f"Epoch {i+1}/{self.epochs} - Hinge Loss: {loss:.4f}, Accuracy: {self.accuracies[-1]:.2f}%, Precision: {self.precisions[-1]:.2f}%, Recall: {self.recalls[-1]:.2f}%, F1 Score: {self.f1_scores[-1]:.2f}%")
                print("------------------------------------------------------------------------")

    def update_weights(self):
        y_label = np.where(self.y <= 0, -1, 1)
        for i in range(self.m):
            s = 0
            for j in range(self.m):
                s += self.alpha[j] * y_label[j] * self.kernel(self.X[j], self.X[i])
            decision_value = y_label[i] * (s + self.b)

            if decision_value >= 1:
                self.alpha[i] -= self.learning_rate * self.lambda_parameter
            else:
                self.alpha[i] += self.learning_rate * (1 - decision_value)
                self.b += self.learning_rate * y_label[i]

    def calculate_hinge_loss(self, X, y):
        y_label = np.where(y <= 0, -1, 1)
        loss = 0
        for i in range(self.m):
            s = 0
            for j in range(self.m):
                s += self.alpha[j] * y_label[j] * self.kernel(self.X[j], X[i])
            margin = y_label[i] * (s + self.b)
            loss += max(0, 1 - margin)
        regularization = (self.lambda_parameter / 2) * np.sum(self.alpha ** 2)
        return loss / self.m + regularization

    def predict(self, X):
        y_hat = []
        y_label = np.where(self.y <= 0, -1, 1)
        for x_i in X:
            result = 0
            for alpha_j, y_j, x_j in zip(self.alpha, y_label, self.X):
                result += alpha_j * y_j * self.kernel(x_j, x_i)
            prediction = result + self.b
            y_hat.append(1 if prediction > 0 else 0)
        return np.array(y_hat)

    def calculate_accuracy(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y_hat == y) * 100

    def calculate_precision(self, X, y):
        y_hat = self.predict(X)
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_positive = np.sum((y_hat == 1) & (y == 0))
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        return precision * 100

    def calculate_recall(self, X, y):
        y_hat = self.predict(X)
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_negative = np.sum((y_hat == 0) & (y == 1))
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        return recall * 100

    def calculate_f1_score(self, X, y):
        precision = self.calculate_precision(X, y)
        recall = self.calculate_recall(X, y)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def get_Weights(self):
        return self.alpha, self.b  # alpha is used instead of w

    def get_stats(self):
        return self.losses, self.accuracies, self.precisions, self.recalls, self.f1_scores
