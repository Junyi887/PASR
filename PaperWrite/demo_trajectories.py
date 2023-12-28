import numpy as np
import matplotlib.pyplot as plt
 
def y(x):
    return 2 * np.exp(-3 * x**2) * np.sin(x) - np.cos(x) * np.sin(0.5 * x**2) + 2 * np.cos(x)**2 + 0.5

def y2(x):
    return 2 * np.exp(-1 * x**2) * np.cos(x) - np.sin(x) * np.cos(0.8 * x**2-x) + 2 * np.sin(x)**2 + 0.5

def y3_new_v3(x):
    return 2.5 * np.sin(4 * x) * np.exp(-0.2 * (x - 5)**2) + 2.0 * np.sin(3 * x) - 0.5 * np.cos(2.5 * x)

def y3(x):
    scaled = (y3_new_v3(x) - -3.8673627398881143) / (3.5519777445185996 - -3.8673627398881143)
    return 0.5 + scaled * (3.0 - 0.5)

def y_new_4(x):
    return 2.0 * np.sin(5 * x) * np.exp(-0.15 * (x - 4)**2) + 1.5 * np.sin(3.5 * x) - 0.7 * np.cos(2.8 * x)

def y4(x):
    scaled = (y_new_4(x) - -3.6303585133319034) / (3.216587177250182 - -3.6303585133319034)
    return y2(x-10)

t = np.linspace(0, 5.5, 11,endpoint=True)
t2 = np.linspace(0, 5.5, 21,endpoint=True)
print(t2[2]-t2[1])
print(t[2]-t[1])
t_ref = np.linspace(0, 5.5, 1001)

# training_label = 'Sampled by dt = 2'
test_label = 'Sampled by dt = 0.5'
first_line = True  # Flag to track the first line
first_dot = True  # Flag to track the first dot


for i,(color_1,color_2,y_func) in enumerate(zip (['r','b','g','m'],['r','b','g','m'],[y,y2,y3,y4])):
    plt.figure(figsize=(7, 2))
    plt.plot(t_ref, y_func(t_ref), f'{color_1}-',alpha = 0.3,linewidth=5)
    plt.ylim(0, 4)
    plt.xticks(np.arange(0, 10.5, 0.5))
    for value in t2:
        if value in t:
            plt.scatter(value, y_func(value), marker='.', s=50, c=color_1)
            # plt.vlines(value, 0, y(value), linestyles='dashed', colors='r')
            training_label = None  # Remove the label after the first plot
        if value in t2 + 0.5 and value not in t:
            plt.scatter(value, y_func(value), marker='.', s=50, c=color_2)
            # plt.vlines(value, 0, y(value), linestyles='dashed', colors='r')
            test_label = None  # Remove the label after the first plot
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Hide y-axis
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # Add x-value labels for each point
    plt.savefig('PaperWrite/paper_figures/demo_dt05'+'.png',bbox_inches='tight',transparent=True)
    plt.savefig(f'PaperWrite/paper_figures/demo_dt05_{color_1}'+'.pdf',bbox_inches='tight',transparent=True)
