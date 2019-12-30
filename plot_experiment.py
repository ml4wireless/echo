import matplotlib
import matplotlib.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from utils.util_data import integers_to_symbols, get_grid_2d


# matplotlib.use('GTKAgg')


def confidence_ellipse(cov, means, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    cov : array-like, shape (2, 2)
        Covariance matrix

    means: array-like, shape (2, )
        Means array

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = means[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = means[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def animated_plot(results_file="/Users/caryn/Desktop/echo/experiments/private_"
                               "preamble/QPSK_poly_hyperparam_search/results/0.npy",
                  results=None):  # (result):
    plt.ion()
    # plt.show()
    if results is None:
        results = np.load(open(results_file, 'rb'), allow_pickle=True)
    meta, results = results[0], results[1:]
    num_agents = meta['num_agents']
    bits_per_symbol = meta['bits_per_symbol']
    num_colors = 2 ** bits_per_symbol
    # cm = plt.get_cmap('tab20')
    # cm = plt.cm.viridis
    # colors = [cm(int(x*cm.N/num_colors)) for x in range(num_colors)]
    cm = plt.get_cmap('tab20')
    colors = [cm(x) for x in range(N)]
    grid = get_grid_2d()
    if num_agents == 2:
        pairs = [(1, 2), (2, 1)]
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 8))
        axes_list = [item for sublist in axes for item in sublist]
    elif num_agents == 1:
        pairs = [(1, 1)]
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 4))
        axes_list = [item for item in axes]
    for result in results:
        _a_ = 0
        for a, b in pairs:
            m, c = (result['mod_std_%i' % a], result['constellation_%i' % a])
            d = result['demod_grid_%i' % b]
            i_means = c[:, 0]
            q_means = c[:, 1]
            cov = np.eye(2, 2) * m ** 2
            i_std, q_std = m
            minor = np.sqrt(9.210) * q_std
            major = np.sqrt(9.210) * i_std
            ax = axes_list[_a_]
            _a_ += 1
            ax.scatter(i_means, q_means, c=colors)
            ells = [Ellipse(xy=c[i], width=major, height=minor, color=colors[i], alpha=.2)
                    for i in range(len(c))]
            # ells = [confidence_ellipse(cov, c[i], ax, facecolor=colors[i], alpha=.2) for i in range(len(c))]
            for i, e in enumerate(ells):
                ax.add_artist(e)
            #     e.set_clip_box(ax.bbox)
            #     e.set_alpha(.2)
            #     e.set_facecolor(colors[i])
            for label, (x, y) in \
                    zip(integers_to_symbols(np.array([i for i in range(2 ** bits_per_symbol)]), bits_per_symbol), c):
                ax.annotate(label,
                            xy=(x, y), xytext=(0, 0), textcoords='offset points')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax1 = axes_list[_a_]
            _a_ += 1
            ax1.scatter(grid[:, 0], grid[:, 1], c=[colors[i] for i in d])
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
        plt.pause(0.0000000001)
        for ax in axes_list:
            ax.clear()
    fig.canvas.draw()

    plt.ioff()
    plt.close(fig)

    # plt.savefig("%s/%s_demod-%d.png" % (plots_dir, "_".join(agent.name.lower().split(" ")), plot_count))
    return
