
import matplotlib.pyplot as plt


def error_plot(errors, title):
    plt.plot(errors, linestyle='-', linewidth=0.6)
    plt.title(title)
    plt.ylim(0, 0.5)
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value Difference')
    plt.ticklabel_format(style='sci', axis='x',
                         scilimits=(0, 0), useMathText=True)
    plt.savefig(f"Plots/title")
    plt.close()
