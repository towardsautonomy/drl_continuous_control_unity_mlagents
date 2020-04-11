import csv
import matplotlib.pyplot as plt
import numpy as np

# log file
log_dir = 'logs/reacher_ddpg/'
log_file = log_dir+'logs.csv'
log_plot_png = log_dir+'plot_test.png'

# moving average
def moving_average(arr, n=5) :
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

# function to plot
def plot():

    # assemble parameters
    episode = []
    score = []
    with open(log_file) as file:
        reader = csv.DictReader( file )
        for line in reader:
            # episode
            episode.append(int(line['episode']))
            # score
            score.append(float(line['score']))

    # get moving average of score
    score_ma = moving_average(score, n=50)
    # plot 
    plt.figure(figsize=(16,9))
    plt.title('Agent Performance', fontweight='bold')
    # training losses and iou
    plt.grid(linestyle='-', linewidth='0.2', color='gray')
    plt.plot(episode, score, 'b-')
    plt.plot(episode, score_ma, 'r-')
    plt.legend(['score','averaged score'], loc='upper left', fancybox=True, framealpha=1., shadow=True, borderpad=1)
    plt.ylabel('Score', fontweight='bold')
    plt.xlabel('Episode', fontweight='bold')

    # plot and save
    plt.savefig(log_plot_png)
    plt.show()

# main function
if __name__ == '__main__':
    plot()