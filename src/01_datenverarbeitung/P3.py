import matplotlib.pyplot as plt


def scatter_plot(xlabel, ylabel, *args):

    for x in args:
        plt.scatter(x[0], x[1], c=x[2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def time_plot(xlabel, ylabel, timestamps, *args):
    for x in args:
        plt.plot(timestamps, x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

