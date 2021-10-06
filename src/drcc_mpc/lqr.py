'''
LQR sub-routine
code by: j.zhu
'''
#%% imports
import numpy as np
import scipy.linalg as linalg


def dt_lqr(A, B, Q, R):
    """
    lqr solver by jz
    using the continuous ricatti eqn solver
    see Bertsekas, ch4.1, p150
    """
    S = linalg.solve_discrete_are(A, B, Q, R)
    # K = 1.0/R *B.T * np.asmatrix(S) #care, not dare
    K = -1*linalg.pinv(B.T @ S @ B + R) @ B.T @ S @ A
    return K, S


if __name__ == '__main__':
    # dynamics
    # X+ = Ax + Bu
    # u \in R, scalar

    # cf. Rawlings ex 3.13
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])

    # cost
    Q = 0.1 * np.eye(2)
    R = 1

    # test step the system

    # give a K matrix
    K, _ = dt_lqr(A, B, Q, R) # note, since u is scalar. dim(K) = 1 * dim(x)

    # clsd loop system
    A_cl = A+B@K

    # %% check stability of cl loop sys
    lmbd = linalg.eigvals(A_cl)
    for l in lmbd:
        print(linalg.norm(l)) # sys is stable iff eigs of A + BK stays within unit circle