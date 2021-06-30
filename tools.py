import numpy as np
import matplotlib.pyplot as plt
import math

def plot_ellipse(covariance, ax, label_t="", linestyle='', alpha_val=0.1, color_def='black', center = [0, 0]):
    w, v = np.linalg.eig(covariance)
    if(np.abs(np.min(w)) < 1e-10):
        e = v[:, np.argmax(w)]
        k = np.sqrt(w[np.argmax(w)])

        p1 = -k * e
        p2 = k * e
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]

        if len(linestyle) > 0:
            if len(label_t) > 0:
                ax.plot(x, y, label=label_t, alpha=alpha_val, color=color_def, linestyle=linestyle)
            else:
                ax.plot(x, y, alpha=alpha_val, color=color_def, linestyle=linestyle)            
        else:
            if len(label_t) > 0:
                ax.plot(x, y, label=label_t, alpha=alpha_val, color=color_def)
            else:
                ax.plot(x, y, alpha=alpha_val, color=color_def) 

    elif covariance.shape[0] == 2:
        x_el = np.array([np.sin(np.linspace(0, 2*math.pi, num=63)), np.cos(np.linspace(0, 2*math.pi, num=63))])
        C = np.linalg.cholesky(covariance)
        y_el = np.dot(C, x_el)
        if len(linestyle) > 0:
            if len(label_t) > 0:
                ax.plot(y_el[0] + center[0], y_el[1] + center[1], label=label_t, alpha=alpha_val, color=color_def, linestyle=linestyle)
            else:
                ax.plot(y_el[0] + center[0], y_el[1] + center[1], alpha=alpha_val, color=color_def, linestyle=linestyle)            
        else:
            if len(label_t) > 0:
                ax.plot(y_el[0] + center[0], y_el[1] + center[1], label=label_t, alpha=alpha_val, color=color_def)
            else:
                ax.plot(y_el[0] + center[0], y_el[1] + center[1], alpha=alpha_val, color=color_def) 