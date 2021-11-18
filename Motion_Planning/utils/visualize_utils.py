import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_path(obs, planned_path_xy, out_path, xlim, ylim, title_str, terrain_img):

    xmin = xlim[0]
    xmax = xlim[1]

    ymin = ylim[0]
    ymax = ylim[1]

    fig = plt.figure()
    # point cloud obstacle
    plt.scatter(obs[:,0], obs[:,1], c='red', s=2)

    linestyle = 'solid'
    color = 'b'

    for path_xy in planned_path_xy:
        path_x = path_xy[0]
        path_y = path_xy[1]

        plt.plot(path_x, path_y, c=color, marker='o', linewidth=4, markersize=5, linestyle=linestyle, alpha=1.0)

    # start
    plt.plot(path_xy[0][0], path_xy[1][0], c='g', marker='o', markersize=6)
    # end
    plt.plot(path_xy[0][-1], path_xy[1][-1], c='y', marker='s', markersize=6)

    ax = fig.gca()
    ax.set_xticks(np.arange(xmin,xmax,50))
    ax.set_yticks(np.arange(ymin,ymax,50))
    ax.set_aspect('equal')
    ax.imshow(terrain_img, cmap='gray')
    plt.xlim(xmin, xmax)
    plt.ylim(ymax, ymin)
    plt.grid()
    plt.title(title_str)

    plt.savefig(out_path, dpi=100, bbox_extra_artists=(lgnd,), bbox_inches='tight')
    plt.close()
