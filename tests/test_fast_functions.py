import numpy as np

import scipy.sparse as sp


from lightfm import _lightfm_fast


def test_in_positives():

    mat = sp.csr_matrix(np.array([[0, 2],
                                  [3, 0]])).astype(np.float32)

    assert _lightfm_fast.__test_in_positives(0, 0, _lightfm_fast.CSRMatrix(mat)) == 0
    assert _lightfm_fast.__test_in_positives(0, 1, _lightfm_fast.CSRMatrix(mat)) == 2

    assert _lightfm_fast.__test_in_positives(1, 0, _lightfm_fast.CSRMatrix(mat)) == 3
    assert _lightfm_fast.__test_in_positives(1, 1, _lightfm_fast.CSRMatrix(mat)) == 0
