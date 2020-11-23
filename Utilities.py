
import matplotlib.pyplot as plt

def plot_error(errors, title, pos):
    plt.figure(pos)
    plt.clf()
    plt.title(title)
    plt.xlabel('# of Iterations')
    plt.ylabel('Q-value Difference')
    plt.ylim(0, 0.5)
    plt.plot(errors, linestyle='-', color='black', linewidth=0.3)
    plt.show()

def error_plot(errors, title):
    plt.plot(errors, linestyle='-', linewidth=0.6)
    plt.title(title)
    plt.ylim(0, 0.5)
    plt.xlabel('Simulation Iteartion')
    plt.ylabel('Q-value Difference')
    plt.ticklabel_format(style='sci', axis='x',
                         scilimits=(0,0), useMathText=True)
    plt.savefig(f"Plots/title")
    plt.close()