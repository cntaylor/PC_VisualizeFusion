import numpy as np
import constant
import numpy.linalg as LA
from numpy.linalg import inv

def mahalanobis_distance(C_c_inv):

    C_ac = inv(inv(constant.C_A) - C_c_inv)
    C_bc = inv(inv(constant.C_B) - C_c_inv)

    C_abc_inv_inv = inv(inv(C_ac) + inv(C_bc))
    C_abc_inv = inv(C_ac + C_bc)

    x_c = (C_abc_inv_inv @ (LA.inv(C_ac) @ constant.MU_A.T + LA.inv(C_bc) @ constant.MU_B.T)).T

    x_ac = (C_ac @ (inv(constant.C_A) @ constant.MU_A.T - C_c_inv @ x_c.T)).T
    x_bc = (C_bc @ (inv(constant.C_B) @ constant.MU_B.T - C_c_inv @ x_c.T)).T

    f = ((x_ac - x_bc) @ C_abc_inv @ (x_ac - x_bc).T)
    return f
