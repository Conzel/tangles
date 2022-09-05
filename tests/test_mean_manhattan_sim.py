import numpy as np
from tangles.cost_functions import BipartitionSimilarity


def test_mean_manhattan_dist():
    points = np.array([[0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]])
    bp = BipartitionSimilarity(points == 1)
    assert np.all(bp.similarities == np.array(
        [[4., 2, 3], [2, 4, 3], [3, 3, 4]]))
    assert bp(np.array([0, 1, 0]) == 1) == (1/(1 * (3 - 1)) * (2 + 3))
    assert bp(np.array([1, 1, 0]) == 1) == (1/(2 * (3 - 2)) * (3 + 3))
