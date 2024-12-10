import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(transforms, trajectory, smoothed_trajectory):
    """Plot video trajectory

    Create a plot of the video's trajectory & smoothed trajectory.
    Separate subplots are used to show the x and y trajectory.

    :param transforms: VidStab transforms attribute
    :param trajectory: VidStab trajectory attribute
    :param smoothed_trajectory: VidStab smoothed_trajectory attribute
    :return: tuple of matplotlib objects ``(Figure, (AxesSubplot, AxesSubplot))``

    >>> from alg1_video_stab import VidStab
    >>> import matplotlib.pyplot as plt
    >>> stabilizer = VidStab()
    >>> stabilizer.gen_transforms(input_path='input_video.mov')
    >>> stabilizer.plot_trajectory()
    >>> plt.show()
    """
    if transforms is None:
        raise AttributeError('No trajectory to plot. '
                             'Use methods: gen_transforms or stabilize to generate the trajectory attributes')

    with plt.style.context('seaborn-v0_8'):
        fig, (ax1, ax2) = plt.subplots(2, sharex='all', figsize=(8, 6))  # Adjust figure size for better layout

        # x trajectory
        ax1.plot(trajectory[:, 0], label='Trajectory', linestyle='-', linewidth=2)
        ax1.plot(smoothed_trajectory[:, 0], label='Smoothed Trajectory', linestyle='--', linewidth=2)
        ax1.set_ylabel('Delta X (dx)', fontsize=12)
        ax1.tick_params(axis='both', labelsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # y trajectory
        ax2.plot(trajectory[:, 1], label='Trajectory', linestyle='-', linewidth=2)
        ax2.plot(smoothed_trajectory[:, 1], label='Smoothed Trajectory', linestyle='--', linewidth=2)
        ax2.set_ylabel('Delta Y (dy)', fontsize=12)
        ax2.tick_params(axis='both', labelsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Unified legend
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=10, frameon=True)

        plt.xlabel('Frame Number', fontsize=12)
        fig.suptitle('Video Trajectory', x=0.15, y=0.98, ha='left', fontsize=14)
        fig.canvas.manager.set_window_title('Trajectory')

        plt.tight_layout(pad=2.0)  # Ensure no overlap
        return fig, (ax1, ax2)


def plot_transforms(transforms, radians=False):
    """Plot stabilizing transforms

    Create a plot of the transforms used to stabilize the input video.
    Plots x & y transforms (dx & dy) in a separate subplot than angle transforms (da).

    :param transforms: VidStab transforms attribute
    :param radians: Should angle transforms be plotted in radians?  If ``false``, transforms are plotted in degrees.
    :return: tuple of matplotlib objects ``(Figure, (AxesSubplot, AxesSubplot))``

    >>> from alg1_video_stab import VidStab
    >>> import matplotlib.pyplot as plt
    >>> stabilizer = VidStab()
    >>> stabilizer.gen_transforms(input_path='input_video.mov')
    >>> stabilizer.plot_transforms()
    >>> plt.show()
    """
    if transforms is None:
        raise AttributeError('No transforms to plot. '
                             'Use methods: gen_transforms or stabilize to generate the transforms attribute')

    with plt.style.context('ggplot'):
        fig, (ax1, ax2) = plt.subplots(2, sharex='all', figsize=(8, 6))  # Adjust figure size for better proportions

        ax1.plot(transforms[:, 0], label='delta x', color='C0', linewidth=2)  # Thicker lines
        ax1.plot(transforms[:, 1], label='delta y', color='C1', linewidth=2)
        ax1.set_ylabel('Delta Pixels', fontsize=12)  # Larger font size
        ax1.tick_params(axis='both', labelsize=10)  # Adjust tick label size
        ax1.grid(True, linestyle='--', alpha=0.6)  # Add subtle gridlines

        if radians:
            ax2.plot(transforms[:, 2], label='delta angle', color='C2', linewidth=2)
            ax2.set_ylabel('Delta Radians', fontsize=12)
        else:
            ax2.plot(np.rad2deg(transforms[:, 2]), label='delta angle', color='C2', linewidth=2)
            ax2.set_ylabel('Delta Degrees', fontsize=12)
        ax2.tick_params(axis='both', labelsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2,
                   labels1 + labels2,
                   loc='upper center',
                   ncol=3,  # Arrange in a row for better aesthetics
                   bbox_to_anchor=(0.5, 1.02),  # Position legend above the plots
                   fontsize=10,
                   frameon=True,
                   edgecolor='black')

        plt.xlabel('Frame Number', fontsize=12)  # Adjust x-label font size
        fig.suptitle('Transformations for Stabilizing', x=0.15, y=0.98, ha='left',
                     fontsize=14)  # Increase title font size
        fig.canvas.manager.set_window_title('Transforms')

        plt.tight_layout(pad=2.0)  # Adjust layout to prevent overlap
        return fig, (ax1, ax2)

