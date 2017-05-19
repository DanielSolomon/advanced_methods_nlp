import numpy as np

def cosine(x, y):
    return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))

def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    nearest_idx = []
    nn = np.apply_along_axis(lambda x: cosine(x, vector), 1, matrix)
    # All cosine values are >= 0, searching for max k is like searching for min k if all values are negated, argpartition can be used and its extremly fast.
    nearest_idx = np.argpartition(-nn, k)[:k]
    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    matrix = np.array([[1, 1], [2, 4], [3, 7], [4, 9]])
    vector = np.array([5, 4])
    assert set(knn(vector, matrix, 3)) == set([0, 1, 3])
    print 'passed.'
    
    
if __name__ == "__main__":
    test_knn()


