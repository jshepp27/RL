import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Simple
# def plot_learning_curve(x_axis, scores, eps_history, filename):
#     fig, ax = plt.subplots()
#     ax.clear()
#
#     plt.plot(x_axis, scores, label="scores")
#     #plt.plot(x_axis, eps_history)
#
#     plt.title("Learning Curve DQN")
#     plt.legend()
#     plt.xlabel("Time")
#     plt.ylabel("Learning")
#
#     plt.savefig(filename)
#     plt.show()

# Extensive
# def plot_learning_curve(x, scores, epsilons, filename):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, label="1")
#     ax2 = fig.add_subplot(111, label="2", frame_on=False)
#
#     ax.plot(x, epsilons, color="C0")
#     ax.set_xlabel("Training Steps", color="C0")
#     ax.set_ylabel("Epsilon", color="C0")
#     ax.tick_params(axis="x", colors="C0")
#     ax.tick_params(axis="y", colors="C0")
#
#     N = len(scores)
#     running_av = np.empty(N)
#     for _ in range(N):
#         running_av[_] - np.mean(scores[max(0, _-100):(_ + 1)])
#
#     ax2.scatter(x, running_av, color="C1")
#     ax2.axes.get_xaxis().set_visible(False)
#     ax2.yaxis.tick_right()
#     ax2.set_ylabel("Score", color="C1")
#     ax2.yaxis.set_label_position("right")
#     ax2.tick_params(axis="y", colors="C1")
#
#     plt.save(filename)

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    # TODO: Noodle this
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
def create_plot(x_axis, scores, eps_history):
    fig, ax = plt.subplots()
    ax.plot(x_axis, scores, label="scores")
    ax.plot(x_axis, eps_history)
    ax.set_title("Learning Curve DQN")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Learning")
    return fig, ax

def create_animation(fig, ax, x_axis, scores, eps_history, filename, writer='ffmpeg', fps=30):
    def animate(i):
        ax.clear()
        ax.plot(x_axis[:i + 1], scores[:i + 1], label="scores")
        ax.plot(x_axis[:i + 1], eps_history[:i + 1])
        ax.set_title("Learning Curve DQN")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Learning")

    ani = animation.FuncAnimation(fig, animate, frames=len(x_axis), repeat=True)
    ani.save(filename, writer=writer, fps=fps)

    plt.show()

    return ani

# if __name__ == "__main__":
#     # Usage example:
#     x_axis = [1, 2, 3, 4, 5]
#     scores = [90, 85, 95, 80, 88]
#     eps_history = [0.1, 0.2, 0.15, 0.3, 0.25]
#     filename = "cartpole_naive_dqn.mp4"
#
#     fig, ax = create_plot(x_axis, scores, eps_history)
#     ani = create_animation(fig, ax, x_axis, scores, eps_history, filename)