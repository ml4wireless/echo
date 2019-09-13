#Utility functions for visualization

import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DPI = 150
HEIGHT = 721
WIDTH = 961

def get_color(k):
    '''
    Returns color given k using the following 16 color map
    '''
    color_map = {
                0: '#1f77b4', # blue
                1: '#ff7f0e', # yellow
                2: '#2ca02c', # green
                3: '#d62728', # red
                # 0: 'deepskyblue',
                # 1: 'orangered',
                # 2: 'm',
                # 3: 'lime',
                4: 'skyblue',
                5: 'lightsalmon',
                6: 'fuchsia',
                7: 'palegreen',
                8: 'lightskyblue',
                9: 'coral',
                10: 'violet',
                11: 'mediumseagreen',
                12: 'steelblue',
                13: 'maroon',
                14: 'hotpink',
                15: 'springgreen'
            }
    return color_map[k%16]


def get_constellation(mod):
        data_si = np.arange(2**mod.bits_per_symbol)
        return mod.modulate(data_si, mode='exploit')

def visualize_constellation(data, labels, data_centers=None, legend_map=None, title_string="", show=True):
    '''
    Plots constellation diagram
    Inputs: 
    data: comple np.array of shape[N] containing I and Q values of symbols
    labels: label, np.array of shape[N] with label for each symbol
    legend_map: map from labels to legends
    title_string: desired title
    '''
    plt.rcParams["figure.figsize"] = [WIDTH/DPI,HEIGHT/DPI]
    plt.rcParams['figure.dpi'] = DPI
    unique_labels = np.unique(labels)
    data = np.array(data)
    labels = np.array(labels)
    plt.ylim ((-3,3))
    plt.xlim ((-3,3))
    if legend_map is None:
        for k,label in enumerate(unique_labels):
            cur_data = data[labels==label]
            plt.scatter(cur_data.real, cur_data.imag, c=get_color(k), alpha=0.5)
            plt.annotate(k, (cur_data[0].real, cur_data[0].imag))
            if data_centers is not None:
                cur_data_centers = data_centers[labels==label]
                plt.scatter(cur_data_centers.real, cur_data_centers.imag,
                            edgecolors='white', facecolor=get_color(k))

        plt.title(title_string)
        if show:
            plt.show()
    else:
        for k,label in enumerate(unique_labels):
            cur_data = data[labels==label]
            plt.scatter(cur_data.real, cur_data.imag, c = get_color(k), label = legend_map[label])
            if data_centers is not None:
                cur_data_centers = data_centers[labels==label]
                plt.scatter(cur_data_centers.real, cur_data_centers.imag,
                            edgecolors='white', facecolor=get_color(k))
        #plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
        plt.title(title_string)
        plt.legend()
        if show:
            plt.show()
        
def visualize_clusters(data, assign, means, legend_map=None, title_string=None):
    '''
    Visualize clusters showing points with corresponding means
    Inputs:
    data: complex array of shape [N] corresponding to I and Q values of each symbol
    assign: integer array of shape [N] corresponding to cluster (mean index) for each symbol
    legend_map: if not None contains cluster labels (eg for QPSK could contain '01', '10', etc.)
    title_string: Desired title
    '''

    if legend_map is None:
        for k in range(len(means)):
            plt.scatter(means[k].real, means[k].imag, marker = '^', s=150, c = get_color(k),edgecolors='black')
            plt.annotate(k, (means[k].real, means[k].imag))
            cur_data = data[assign==k]
            plt.scatter(cur_data.real, cur_data.imag, c = get_color(k), alpha = 0.15)
        if title_string is not None:          
            plt.title(title_string)
        plt.show()
    else:
        for k in range(len(means)):
            plt.scatter(means[k].real, means[k].imag, marker = '^', s=150, c = get_color(k),edgecolors='black')
            plt.annotate(legend_map[k], (means[k].real, means[k].imag))
            cur_data = data[assign==k]
            plt.scatter(cur_data.real, cur_data.imag, c = get_color(k), alpha = 0.15)
        if title_string is not None: 
            plt.title(title_string)
        plt.show() 

def gen_demod_grid(points_per_dim=10, min_val=-3.5, max_val=3.5):
    grid_1d = np.linspace(min_val,max_val,points_per_dim) 
    grid_2d = np.squeeze(np.array(list(itertools.product(grid_1d, grid_1d))))
    data_c = grid_2d[:,0] + 1j*grid_2d[:,1]
    return {'data':data_c, 'min_val':min_val, 'max_val':max_val, 'points_per_dim':points_per_dim}

def get_grid(demodulator, points_per_dim=10, min_val=-3.5, max_val=3.5):
        grid = gen_demod_grid(points_per_dim=points_per_dim, min_val=min_val, max_val=max_val)
        data_c = grid['data']
        labels = demodulator.demodulate(data_c, mode='exploit').astype(np.uint8)
        ret = dict(grid)
        ret.pop('data')
        ret['labels'] = labels
        return ret

def visualize_decision_boundary(demod, points_per_dim=10, title_string=None, legend_map=None, show=True):
        '''
        Visualize decision boundary for demodulation by passing grid of 2d plane as input and demodulating 
        in mode 'exploit'
        Returns
        num_constellation_points: Number of unique constellation points in grid
        '''
        #Generate grid
        data_c = gen_demod_grid(points_per_dim=points_per_dim)['data']
        labels_si_g= demod.demodulate(data_c=data_c, mode = 'exploit')
        unique_labels_si_g = np.unique(labels_si_g)
        def plot_func():
            if legend_map is None:
                for i in range(unique_labels_si_g.shape[0]):
                    cur_label = unique_labels_si_g[i]
                    cur_data_c = data_c[labels_si_g == cur_label]
                    plt.scatter(cur_data_c.real, cur_data_c.imag, s=15, color=get_color(cur_label))
                    plt.annotate(cur_label, (cur_data_c[cur_data_c.shape[0]//2].real, cur_data_c[cur_data_c.shape[0]//2].imag))

            else:
                for i in range(unique_labels_si_g.shape[0]):
                    cur_label = unique_labels_si_g[i]
                    cur_data_c = data_c[labels_si_g == cur_label]
                    plt.scatter(cur_data_c.real, cur_data_c.imag, s=15, color=get_color(cur_label), label=legend_map[cur_label])
                plt.legend()
                #leg = plt.legend(loc = 'upper right',bbox_to_anchor=(1.1, 1.05))
                #for h in range(len(leg.legendHandles)):
                #    leg.legendHandles[h]._sizes = [35]

            if title_string is not None:
                plt.title(title_string)

            if show:
                plt.show()

            num_constellation_points = unique_labels_si_g.shape[0]
            return num_constellation_points
        return plot_func
        
class PlotManager:
    '''
    Helps track and manage plotting data so that mutiple agents can contribute to a single subplot.
    '''
    def __init__(self, save_dir=None, rows=1, cols=1):
        self.save_dir = save_dir
        self.rows = rows #number of rows of plots
        self.cols = cols #number of columns of plots
        self.args = [None for _ in range(rows*cols)] #list of data used for generating the subplots
        self.accept_plots = [False for _ in range(rows)] #set to True when ready to plot (e.g., when (num_steps % visualize_every) == 0)
        
        self.times_plotted = 0
    
    def add_plot(self, func, args, row=1, col=1):
        '''
        func: generates the plot
        args: arguments needed for func
        row: row in subplot to put this plot
        col: column in subplot to put this plot
        '''
        args["plot_func"] = func
        self.args[(row-1)*self.cols+col-1] = args
        if None not in self.args:
            self.plot_all()
        
    def plot_all(self):
        '''
        Do not need to call this. The function is called automatically when all
        subplots have been added. (see add_plot)
        '''
        plt.figure(figsize=(7*self.cols,5*self.rows))
        for i in range(self.rows*self.cols):
            if self.args[i] is not None:
                plt.subplot(self.rows, self.cols, i+1)
                plot_func = self.args[i].pop("plot_func")
                plot_func(**self.args[i])
        if self.save_dir is not None:
            plt.savefig("%s/plots/plot_%d.pdf"%(self.save_dir, self.times_plotted+1))
        else:
            plt.show()
        self.times_plotted += 1
        print("<"+"-"*50+">")
        self.args = [None for _ in range(self.rows*self.cols)]
        self.accept_plots = [False for _ in range(self.rows)]
    
    def turn_plot_on(self, row):
        self.accept_plots[row-1] = True
    
    def is_plot_on(self, row):
        return self.accept_plots[row-1]

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
