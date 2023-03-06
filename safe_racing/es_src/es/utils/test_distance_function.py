import matplotlib.pyplot as plt
from es.utils.utils import *


def draw_pts(pts, ax, c='b', mksize=5.0, label=None, pointonly=False, linewidth=0.5):
    """
    pts: (2, n)
    """
    x = pts[0]
    y = pts[1]
    if pointonly:
        ax.plot(x, y, 'o', c=c, markersize=mksize, label=label)
    else:
        ax.plot(x, y, c=c, markersize=mksize, label=label, linewidth=linewidth)


def main():
    origin = np.array((0, 0))
    pose_1 = np.array([0.0, 1.0, 0.0])
    pose_2 = np.array([1.0, 0.0, 0.0])
    # pose_1 = np.random.random((3))
    # pose_2 = np.random.random((3))
    length = 0.6
    width = 0.3
    vertice_1 = get_vertices(pose_1, length, width)
    vertice_2 = get_vertices(pose_2, length, width)
    D = pose_2[:2] - pose_1[:2]
    # D = pose_1[:2] - pose_2[:2]
    dist, d = distance_debug(vertice_1, vertice_2, D)
    print(f'distance is {dist}')

    fig, ax = plt.subplots(figsize=(20, 10))
    draw_pts(vertice_1.T, ax, c='r', pointonly=True)
    draw_pts(vertice_2.T, ax, c='g', pointonly=True)
    # draw_pts(np.vstack((p1, origin)).T, ax, mksize=1.0, linewidth=2.0, c=(0.6, 0.3, 0.4, 1.0), label='p1')
    # draw_pts(np.vstack((p2, origin)).T, ax, mksize=1.0, linewidth=2.0, c=(0.7, 0.2, 0.5, 1.0), label='p2')
    draw_pts(np.vstack((d, origin)).T, ax, mksize=1.0, linewidth=2.0, c=(0.8, 0.5, 0.3, 1.0), label='d')
    draw_pts(np.vstack((pose_1[:2], pose_2[:2])).T, ax, linewidth=1.0, c='k')

    ax.axis('equal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


