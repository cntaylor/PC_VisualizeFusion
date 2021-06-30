import numpy as np
import numpy.linalg as LA

PI = np.pi
DIMS = 2

STEP = PI/50

# C_A = np.array([[0.59545192, 0.72251469],
#                 [0.72251469, 4.30814793]])

C_A = np.diag([4, 1])

C_A_INV = LA.inv(C_A)

# MU_A = np.array([-0.3, -0.3])
MU_A = np.array([0.1, .5])


# C_B = np.array([[3.21247396, 1.60506457],
#                 [1.60506457, 1.55566236]])

C_B = np.diag([1, 2])

C_B_INV = LA.inv(C_B)

MU_B = np.array([0.1, -0.14])

K_INIT = 0.1

I = 1e-10*np.identity(2)