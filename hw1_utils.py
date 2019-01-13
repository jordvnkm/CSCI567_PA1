import math
import numpy
from typing import List

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

    return (2 * numerator) / (real_total + predicted_total)

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


if __name__ == "__main__":
    print(gaussian_kernel_distance([1, 3], [3, 5]))
