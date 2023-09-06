import proxsuite
import numpy as np

def init_prosuite_qp(n_v, n_eq, n_in):
    qp = proxsuite.proxqp.dense.QP(n_v, n_eq, n_in)
    # Randomly initialize the QP
    qp.init(np.eye(n_v), None, None, None, None, None, None)
    qp.settings.eps_abs = 1.0e-6
    qp.settings.max_iter = 20
    qp.solve()
    return qp