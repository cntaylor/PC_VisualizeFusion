import numpy as np
import matplotlib.pyplot as plt
import constant
import formulas
from scipy.stats import chi2
import numpy.linalg as LA
import tools
from scipy.stats import chi2

def get_critical_value(dimensions, alpha):
    return chi2.ppf((1 - alpha), df=dimensions)


ax = plt.axes()

def generate_C_c_inv(k, theta):
    R = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])

    D = np.zeros((2, 2))
    D[0][0] = k
    return R @ D @ R.T



def binary_search(theta, lo = constant.K_INIT, hi = 1):
    eta = get_critical_value(1, 0.05)
    k = (lo + hi)/2
    max_dist = formulas.mahalanobis_distance(generate_C_c_inv(hi*.99, theta))
    if(max_dist < eta):
        return hi
    while lo < hi:
        k = (lo + hi)/2
        k_dist = formulas.mahalanobis_distance(generate_C_c_inv(k*.99, theta))
        if eta - k_dist > constant.TOL:
            lo = k
        elif k_dist - eta > constant.TOL:
            hi = k
        else:
            break
    print(hi)
    print(k)
    print("=================")
    return k



def main_perform_fusion(theta):
    
    #INITIALIZATION

    #define rotation matrices

    C_c_inv = generate_C_c_inv(constant.K_INIT, theta)
    w, v = np.linalg.eig(C_c_inv)

    e = v[:, np.argmax(w)]


    #upper bound of k
    u = 1/max(e.T @ constant.C_A @ e, e.T @ constant.C_B @ e)

    return binary_search(theta, hi = u), u


if __name__ == "__main__":
    theta = np.arange(0, constant.PI, constant.STEP)
    tools.plot_ellipse(constant.C_A, ax)
    tools.plot_ellipse(constant.C_B, ax)
    tools.plot_ellipse(constant.C_A_INV, ax, color_def="blue", alpha_val=1)
    tools.plot_ellipse(constant.C_B_INV, ax, color_def="blue", alpha_val=1)
    plt.grid(True)
    ax.set_aspect('equal')
    for t in theta:
        k, u = main_perform_fusion(t)
        C_c_inv = generate_C_c_inv(k, t)
        C_c_inv_2 = generate_C_c_inv(u, t)
        # tools.plot_ellipse(C_c_inv_2, ax, color_def='green', alpha_val=1)
        tools.plot_ellipse(C_c_inv, ax, color_def='red', alpha_val=1)
    plt.show()
