
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.base import BaseEstimator, ClassifierMixin



class IMBHN(BaseEstimator, ClassifierMixin):
    def __init__(self, c_rate=0.1, n_iterations=1000, error=0.01, weights='random'):
        self.c_rate = c_rate
        self.n_iterations = n_iterations
        self.error = error
        self.weights = weights

    #@profile
    def fit(self, X, y):
        num_it = 0
        #n_class = len(set(y))
        mean_error = float('inf')

        # one-hot encoding
        self.terms = sorted(set(y))
        hash = dict(zip(self.terms, range(len(self.terms))))
        self.hash2term = dict(zip(range(len(self.terms)), self.terms))
        new_y = [hash[label] for label in y]

        self.n_train = len(y)
        self.n_classes = len(self.terms)
        self.n_terms = len(X[0])

        np.random.seed(2)

        self.fTerms = np.zeros((self.n_terms, self.n_classes))
        #self.fTerms = np.random.random((self.n_terms, self.n_classes))

        # priori likelihood
        """
        totals = np.zeros(self.n_terms)
        for i in range(self.n_terms):
            totals[i] = (X[:,i] > 0).sum()

        for term in range(self.n_terms):
            for cl in range(self.n_classes):
                self.fTerms[term][cl] = (X[y == cl][:, term] > 0).sum()/totals[term]
        """

        self.fDocs  = np.zeros((self.n_train, self.n_classes))
        for i in range(self.n_train):
            self.fDocs[i][new_y[i]] = 1

        adj_list = [[] for i in range(self.n_train)]


        for inst in range(self.n_train):
            for term in range(self.n_terms):
                if X[inst][term]:
                    adj_list[inst].append((term, X[inst][term]))

        while num_it < self.n_iterations and mean_error > self.error:
            mean_error = 0
            for inst in range(self.n_train):
                neighbors = adj_list[inst]
                estimated = self.classify(neighbors)
                #print('max',X[inst].max())
                for classe in range(self.n_classes):
                    #print('nei',neighbors)
                    #print('classe',classe)
                    #print(estimated)
                    #print(self.fDocs[inst][classe])
                    #print(estimated[classe])
                    error = self.fDocs[inst][classe] - estimated[classe]
                    mean_error += (error*error)/2
                    for term in neighbors:
                        current_weight = self.fTerms[term[0]][classe]
                        new_weight = current_weight + (self.c_rate * term[1] * error)
                        self.fTerms[term[0]][classe] = new_weight
            num_it += 1
            mean_error = mean_error/self.n_train


    def predict(self, X):
        """
        y = []
        for x in X:
            out = np.dot(x, self.F)
            argmax_out = np.zeros(out.shape)
            argmax_out[np.argmax(out)] = 1
            #argmax_out[np.arange(out.shape[0]), np.argmax(out, axis=1)] = 1

            y.append(self.hash2term[argmax_out.argmax()])
        return y
        """
        adj_list = [[] for i in range(len(X))]

        for inst in range(len(X)):
            for term in range(self.n_terms):
                if X[inst][term] > 0:
                    adj_list[inst].append((term, X[inst][term]))
        y = []
        for inst in range(len(X)):
            neighbors = adj_list[inst]
            y.append(self.hash2term[self.classify(neighbors).index(1)])

        return y

    def score(self, X, y):
        y_pred = self.predict(X)

        return accuracy_score(y,y_pred)


    def classify(self, neighbors):
        #print(neighbors)
        classes = []
        total = 0
        for term in neighbors:
            total += term[1]

        if not total:
            classes = [0] * self.n_classes
            classes[0] = 1
            return classes

        for classe in range(self.n_classes):
            acm = 0
            for term in neighbors:
                acm += term[1] * self.fTerms[term[0]][classe]
            classes.append(acm)

        index_max = classes.index(max(classes))

        classes = [0] * len(classes)
        classes[index_max] = 1

        return classes
