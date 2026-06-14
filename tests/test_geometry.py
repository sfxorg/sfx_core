from geometry.sem_nodes import get_sem_diff_matrix_2d

def test_sem_matrix():
    nodes, D = get_sem_diff_matrix_2d(4)
    assert D.shape == (5, 5)
