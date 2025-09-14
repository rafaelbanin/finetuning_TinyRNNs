from matplotlib import pyplot as plt
from plotting import plot_start
import numpy as np
# Initialize the starting point
Q_L = Q_R = 0
alpha = 0.5
n_updates = 5

trajectory_points = [(Q_L, Q_R)]

def update(Q, alpha):
    return Q + alpha * (1 - Q)


# Plotting
if False:
    for _ in range(n_updates):
        Q_L = update(Q_L, alpha)
        trajectory_points.append((Q_L, Q_R))  # Record the point after Q_L update
        Q_R = update(Q_R, alpha)
        trajectory_points.append((Q_L, Q_R))  # Record the point after Q_R update
    size = 3
    plot_start(figsize=(2, 2))
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.plot([p[0] for p in trajectory_points[:-1]], [p[1] for p in trajectory_points[:-1]], '-o', color='gray', alpha=1, markersize=size)
    plt.plot(trajectory_points[0][0], trajectory_points[0][1], color='red', marker='o', markersize=size)
    plt.plot(trajectory_points[1][0], trajectory_points[1][1], color='orange', marker='o', markersize=size)
    
    plt.plot(trajectory_points[-3][0], trajectory_points[-3][1], color='blue', marker='o', markersize=size)
    plt.plot(trajectory_points[-2][0], trajectory_points[-2][1], color='green', marker='o', markersize=size)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel('$Q_L$')
    plt.ylabel('$Q_R$')
    plt.grid(True)
    plt.savefig('../figures/revision_diagram_2d_to_1d.pdf', bbox_inches='tight')
    plt.show()

if True:
    fig, ax = plot_start(figsize=(2,2))

    from hilbertcurve.hilbertcurve import HilbertCurve
    n_iter =4
    dim = 2
    n_interval_x = 2**n_iter # Number of intervals in x-axis
    num_points = 2**(n_iter*dim)
    hilbert_curve = HilbertCurve(n_iter, dim)
    distances = list(range(num_points))
    points = np.array(hilbert_curve.points_from_distances(distances))
    points= points / points.max() # Normalize to [0, 1]
    points = points * (n_interval_x - 1)/n_interval_x + 1/(2*n_interval_x) # normalize to [1/(2*n_interval_x), 1-1/(2*n_interval_x)]
    # for point, dist in zip(points, distances):
    #     print(f'point(h={dist}) = {point}')
    plt.plot([p[0] for p in points], [p[1] for p in points], '-')

    size = 1
    alpha=0.465
    Q_L, Q_R = (0.03, 0.08)
    trajectory_points = [(Q_L, Q_R)]
    Q_L = update(Q_L, alpha)
    trajectory_points.append((Q_L, Q_R))
    plt.plot([p[0] for p in trajectory_points], [p[1] for p in trajectory_points], '-o', color='gray',
             alpha=1, markersize=size, linewidth=0.5)
    plt.plot(trajectory_points[0][0], trajectory_points[0][1], color='red', marker='o', markersize=size)
    plt.plot(trajectory_points[1][0], trajectory_points[1][1], color='orange', marker='o', markersize=size)

    Q_L, Q_R = (0.09, 0.087)
    trajectory_points = [(Q_L, Q_R)]
    Q_L = update(Q_L, alpha)
    trajectory_points.append((Q_L, Q_R))
    plt.plot([p[0] for p in trajectory_points], [p[1] for p in trajectory_points], '-o', color='gray',
             alpha=1, markersize=size, linewidth=0.5)
    plt.plot(trajectory_points[0][0], trajectory_points[0][1], color='blue', marker='o', markersize=size)
    plt.plot(trajectory_points[1][0], trajectory_points[1][1], color='green', marker='o', markersize=size)
    if n_iter == 3:
        plt.xticks(np.arange(0, 1+1/8, 1/8), ['0', '', '0.25', '', '0.5', '', '0.75', '', '1'])
        plt.yticks(np.arange(0, 1+1/8, 1/8), ['0', '', '0.25', '', '0.5', '', '0.75', '', '1'])
    elif n_iter == 4:
        plt.xticks(np.arange(0, 1+1/16, 1/16), ['0', '', '', '', '0.25', '', '', '', '0.5', '', '', '', '0.75', '', '', '', '1'])
        plt.yticks(np.arange(0, 1+1/16, 1/16), ['0', '', '', '', '0.25', '', '', '', '0.5', '', '', '', '0.75', '', '', '', '1'])
    plt.xlabel('$Q_L$')
    plt.ylabel('$Q_R$')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    major_ticks = np.arange(0, 101, 20)
    minor_ticks = np.arange(0, 101, 5)

    ax.grid(which='both')
    ax.grid(which='major', alpha=0.5)
    plt.savefig(f'../figures/revision_diagram_2d_to_1d_hilbert_{n_iter}.pdf', bbox_inches='tight')

    plt.show()