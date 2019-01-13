from __future__ import division, print_function

from typing import List, Callable

import numpy as np
import scipy
from collections import Counter


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################
import heapq

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function
        self.training_examples = {}

    def train(self, features: List[List[float]], labels: List[int]):
        for index in len(features):
            training_examples[features[index]] = labels[index] 
        
    
    def predict(self, features: List[List[float]]) -> List[int]:
        predicted_classes = []
        for index in len(features):
            k_nearest = get_k_neighbors(features[index])
            predicted_classes.append(_get_class_from_neighbors(k_nearest))
        return predicted_classes

    def _get_class_from_neighbors(k_nearest):
        counter = Counter(k_nearest)
        label, count = counter.most_common()[0]
        return label
        
        
    # returns the labels for the k nearest neighbors
    def get_k_neighbors(self, point):
        k_nearest = []
        distances_to_features = {} 
        distances = set()
        for features in self.training_examples.keys():
            distance = self.distance_function(point, features)
            feature_list = []
            if distance in distances_to_features:
                feature_list = distance_to_features[distance]
            feature_list.append(features)

            distances.add(distance)
            distances_to_features[distance] = feature_list

        smallest_distances = heapq.nsmallest(self.k, distances)
        smallest_distances.sort()
        for distance in smallest_distances:
            feature_list = distances_to_features[distance]
            for feature in feature_list:
                if len(k_nearest == self.k):
                    return k_nearest
                k_nearest.append(self.training_examples[feature])
        return k_nearest

    # Gets the f1 score from given knn_classifier and f1_score function.
    def _get_f1_score_from_classifier(knn_classifier, Xval, yval, f1_score):
        predictions = []
        for features in Xval:
            predictions.append(knn_classifier.predict(features))
        score = f1_score(yval, predictions)
        return score
            

        
    # model selection function where you need to find the best k     
    def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, f1_score, Xval, yval, Xtest, ytest):
        model = None
        best_k = 0
        current_k = 0
        best_f1_score = None

        while current_k < len(ytrain):
            current_k += 1
            for distance_function in distance_funcs:
                knn_classifier = KNN(current_k, distance_function)
                knn_classifier.train(Xtrain, ytrain)
                knn_score = _get_f1_score_from_classifier(knn_classifier, Xval, yVal)
                if best_f1_score == None or best_f1_score < knn_score:
                    best_f1_score = knn_score
                    best_k = current_k
                    model = knn_classifier
                

        
        #Dont change any print statement
        print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) + 
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

        print()
        print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
              'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
        print()

        return best_k, model
    
    #TODO: Complete the model selection function where you need to find the best k with transformation
    def model_selection_with_transformation(distance_funcs,scaling_classes, Xtrain, ytrain, f1_score, Xval, yval, Xtest, ytest):
        
                #Dont change any print statement
                print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
                          'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) + 
                          'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
    
                print()
                print('[part 1.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
                      'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
                print()
        
        
    #TODO: Do the classification 
    def test_classify(model):
        

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)

