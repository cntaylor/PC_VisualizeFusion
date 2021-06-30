import numpy as np
import matplotlib.pyplot as plt
import constant
import formulas
from scipy.stats import chi2
import numpy.linalg as LA
import tools

ax = plt.axes()

def generate_C_c_inv(k, theta):
    R = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])

    D = np.zeros((2, 2))
    D[0][0] = k
    return R @ D @ R.T



def binary_search(theta, lo = constant.K_INIT, hi = 1):
    # mid = (lo + hi)/2
    # while lo < hi:
    #     k = (lo + hi)/2
    #     k_dist = formulas.mahalanobis_distance(generate_C_c(k, theta))
    #     if

    # print(generate_C_c_inv(hi, theta))
    print('theta',theta,'mah distance',formulas.mahalanobis_distance(generate_C_c_inv(hi*.99, theta)))
    # print(formulas.mahalanobis_distance(generate_C_c_inv(lo, theta)))
    # print(chi2.ppf((1 - 0.05), df=2))
 
    # C_ac = LA.inv(LA.inv(constant.C_A) - LA.inv(generate_C_c(lo, theta)))
    C_c_inv = generate_C_c_inv(hi*.99, theta)
    ax = plt.gca()
    tools.plot_ellipse(C_c_inv, ax,color_def='red', alpha_val=1)
    tools.plot_ellipse(constant.C_A_INV,ax,color_def='blue', alpha_val=1)
    tools.plot_ellipse(constant.C_B_INV,ax,color_def='blue', alpha_val=1)
    print ('C_ac_inv is',constant.C_A_INV-C_c_inv)
    C_ac = LA.inv(constant.C_A_INV-C_c_inv)
    tools.plot_ellipse(C_ac,ax,color_def='green', alpha_val=1)
    print ('C_bc_inv is',constant.C_B_INV-C_c_inv)
    C_bc = LA.inv(constant.C_B_INV-C_c_inv)
    tools.plot_ellipse(C_bc,ax,color_def='green', alpha_val=1)
    plt.title(f'theta of {theta}')
    plt.show()
    
    # tools.plot_ellipse(C_ac, ax, color_def='green')
    # plt.show()

    k_s = np.linspace(0, .99*hi)
    maha = []
    for k in k_s:
        maha.append(formulas.mahalanobis_distance(generate_C_c_inv(k, theta)))

    plt.plot(k_s, maha)
    plt.show()    



def main_perform_fusion(theta):
    
    #INITIALIZATION

    #define rotation matrices

    C_c_inv = generate_C_c_inv(constant.K_INIT, theta)
    w, v = np.linalg.eig(C_c_inv)

    e = v[:, np.argmax(w)]


    #upper bound of k
    u = 1/max(e.T @ constant.C_A @ e, e.T @ constant.C_B @ e)
    print(u)

    binary_search(theta, hi = u)


if __name__ == "__main__":
    theta = np.arange(0, constant.PI, constant.STEP)
    tools.plot_ellipse(constant.C_A, ax)
    tools.plot_ellipse(constant.C_B, ax)
    tools.plot_ellipse(constant.C_A_INV, ax, color_def="blue", alpha_val=1)
    tools.plot_ellipse(constant.C_B_INV, ax, color_def="blue", alpha_val=1)
    plt.grid(True)
    ax.set_aspect('equal')
    for t in theta:
        main_perform_fusion(t)
    print("Original Mahalanobis distance is:",formulas.mahalanobis_distance(np.zeros((2,2))))
    plt.show()
