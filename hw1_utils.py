import math
import numpy
from typing import List

# Information Gain function
def Information_Gain(s, branches):
    # S: float - entropy of current state
    # branches: List[List[any]]
    # return: float
    total_examples = 0
    for branch in branches:
        total_examples += numpy.sum(branch)

    if total_examples == 0:
        return 0

    remainder = 0
    for branch in branches:
        probability = numpy.sum(branch) / total_examples
        entropy = get_entropy(branch)
        remainder += probability * entropy

    return s - remainder

def get_entropy(branch):
    total = numpy.sum(branch)
    if total == 0:
        return 0
    answer = 0
    for class_count in branch:
        probability = class_count / total
        answer += probability * math.log(probability, 2)  
    return answer * -1



# implement reduced error pruning
def reduced_error_pruning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List[any]
    pass


# print current tree
# Do not change this function
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    for idx_cls in range(node.num_cls):
        string += str(node.labels.count(idx_cls)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    numerator = 0.0
    real_total = 0.0
    predicted_total = 0.0
    
    for index in range(0, len(real_labels)):
        numerator += real_labels[index] * predicted_labels[index]
        real_total += real_labels[index]
        predicted_total += predicted_labels[index]

    if real_total + predicted_total == 0:
        return 0

    return (2 * numerator) / (real_total + predicted_total)

def f1_score2(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    if len(real_labels) == 0:
        return 0
    possible_labels = get_possible_labels(real_labels)

    total_f1_score = 0
    for label in possible_labels:
        true_positives = get_true_positives(real_labels, predicted_labels, label)
        true_negatives = get_true_negatives(real_labels, predicted_labels, label)
        false_positives = get_false_positives(real_labels, predicted_labels, label)
        false_negatives = get_false_negatives(real_labels, predicted_labels, label)

        precision = None
        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives) 

        recall = None
        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives) 

        if precision + recall != 0:
            total_f1_score += (2 * precision * recall) / (precision + recall)

    return total_f1_score / len(possible_labels) 

def get_true_positives(real_labels, predicted_labels, target_label):
    true_positives = 0
    for index in range(0, len(real_labels)):
        if real_labels[index] == target_label and predicted_labels[index] == target_label:
            true_positives += 1
    return true_positives

def get_true_negatives(real_labels, predicted_labels, target_label):
    true_negatives = 0
    for index in range(0, len(real_labels)):
        if predicted_labels[index] != target_label and real_labels[index] != target_label:
            true_negatives += 1
    return true_negatives

def get_false_positives(real_labels, predicted_labels, target_label):
    false_positives = 0
    for index in range(0, len(real_labels)):
        if predicted_labels[index] == target_label and real_labels[index] != target_label:
            false_positives += 1
    return false_positives

def get_false_negatives(real_labels, predicted_labels, target_label):
    false_negatives = 0
    for index in range(0, len(real_labels)):
        if predicted_labels[index] != target_label and real_labels[index] == target_label:
            false_negatives += 1
    return false_negatives

def get_possible_labels(real_labels: List[int]) -> List[int]:
    possible_labels = set()
    for label in real_labels:
        possible_labels.add(label)
    return list(possible_labels)

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    total = 0.0
    for index in range(0, len(point1)):
        total += (point1[index] - point2[index]) ** 2
    return math.sqrt(total)


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    total = 0.0
    for index in range(0, len(point1)):
        total += point1[index] * point2[index]
    return total


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    total = 0.0
    for index in range(0, len(point1)):
        total += (point1[index] - point2[index]) ** 2
    return math.exp(total * 0.5) * -1.0



def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    total = 0.0
    point1Norm = 0.0
    point2Norm = 0.0

    for index in range(0, len(point1)):
        total += point1[index] * poin2[index]
        point1Norm += point1[index] ** 2
        point2Norm += point2[index] ** 2

    point1Norm = math.sqrt(point1Norm)
    point2Norm = math.sqrt(point2Norm)
    return total / (point1Norm * point2Norm)

def normalization_scaler(features: List[List[float]]) -> List[List[float]]:
    scaled_features = []
    for feature in features:
        denominator = 0
        for index in range(0, len(feature)):
            denominator += feature[index] ** 2
        denominator = math.sqrt(denominator)
        new_feature = []
        for index in range(0, len(feature)):
            if denominator == 0:
                new_feature.append(0)
            else:
                new_feature.append(feature[index] / denominator)
        scaled_features.append(new_feature)
    return scaled_features

class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.
    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).
    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]
        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]
        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]
        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """

    def __init__(self):
        self.max_list = None 
        self.min_list = None
        

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        if self.max_list == None and self.min_list == None:
            self._create_max_min_list(features)

        scaled_features = []
        for feature in features:
            scaled_feature = []
            for index in range(0, len(feature)):
                max_val = self.max_list[index]
                min_val = self.min_list[index]
                denominator = max_val - min_val
                if denominator == 0:
                    scaled_value = 0
                else:
                    scaled_value = (feature[index] - min_val) / (max_val - min_val)
                scaled_feature.append(scaled_value)
            scaled_features.append(scaled_feature)
        return scaled_features
                


    def _create_max_min_list(self, features: List[List[float]]):
        for feature in features:
            for index in range(0, len(feature)):
                if self.max_list == None and self.min_list == None:
                    self.max_list = feature.copy() 
                    self.min_list = feature.copy()
                    break
                else:
                    self.max_list[index] = max(self.max_list[index], feature[index])
                    self.min_list[index] = min(self.min_list[index], feature[index])



if __name__ == "__main__":
    branches = [[5, 2],[3, 10]]
    print(str(Information_Gain(0.97, branches)))
