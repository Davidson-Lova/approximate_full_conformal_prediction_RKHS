import matplotlib.pyplot as plt
import numpy as np


def two_group_boxplot(
    data_a, data_b, labels, title, ticks=None, ylim=None, yscale="linear", savePath=None
):
    if ticks is None:
        ticks = list(range(len(data_a)))

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)

    # plt.figure()

    fig, ax = plt.subplots()

    bpl = ax.boxplot(
        data_a, positions=np.array(range(len(data_a))) * 2.0 - 0.4, sym="", widths=0.6
    )
    bpr = ax.boxplot(
        data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym="", widths=0.6
    )
    set_box_color(bpl, "#D7191C")  # colors are from http://colorbrewer2.org/
    set_box_color(bpr, "#2C7BB6")

    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c="#D7191C", label=labels[0])
    ax.plot([], c="#2C7BB6", label=labels[1])
    ax.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    ax.grid()
    plt.yscale(yscale)
    plt.ylim(None if ylim is None else ylim)
    plt.title(title)
    if savePath is None:
        plt.close(fig)
    else:
        plt.savefig(savePath, bbox_inches="tight")
        plt.close(fig)


def one_group_boxplot(
    data, labels, title, ticks=None, ylim=None, yscale="linear", savePath=None
):
    if ticks is None:
        ticks = list(range(len(data)))

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)

    fig, ax = plt.subplots()

    bpl = ax.boxplot(
        data, positions=np.array(range(len(data))) * 2.0 - 0.4, sym="", widths=0.6
    )
    set_box_color(bpl, "#D7191C")  # colors are from http://colorbrewer2.org/

    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c="#D7191C", label=labels[0])
    ax.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    ax.grid()
    plt.yscale(yscale)
    plt.ylim(None if ylim is None else ylim)
    plt.title(title)
    if savePath is None:
        plt.close(fig)
    else:
        fig.savefig(savePath, bbox_inches="tight")
        plt.close(fig)


def one_boxplot(
    data, labels, title, ticks=None, ylim=None, yscale="linear", savePath=None
):
    if ticks is None:
        ticks = list(range(len(data)))

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)

    fig, ax = plt.subplots()

    bpl = ax.boxplot(data)
    set_box_color(bpl, "#D7191C")  # colors are from http://colorbrewer2.org/

    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c="#D7191C", label=labels[0])
    ax.legend()
    ax.grid()
    plt.yscale(yscale)
    plt.ylim(None if ylim is None else ylim)
    plt.title(title)
    if savePath is None:
        plt.close(fig)
    else:
        plt.savefig(savePath, bbox_inches="tight")
        plt.close(fig)
