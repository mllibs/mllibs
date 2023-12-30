
from mllibs.nlpi import nlpi
from mllibs.dict_helper import sfp,sfpne
import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from arch.bootstrap import IIDBootstrap


'''

Visualise Statistical Differences

'''

class stats_plot(nlpi):
    
    def __init__(self):
        self.name = 'stats_plot'  

        path = pkg_resources.resource_filename('mllibs','/stats/mstats_plot.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)

        # Colour Palettes
        # default_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf" ] # old plotly palette
        # default_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'] # express new
        # default_colors = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477','#66AA00','#B82E2E','#316395'] # g10
        default_colors_p = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252'] # my custom (plotly)
        default_colors_s = ['#568f8b','#b4d2b1', '#1d4a60', '#cd7e59', '#ddb247', '#d15252'] # my custom (seaborn)
        self.default_colors = [default_colors_p,default_colors_s] # to keep the order synchronised
        
    # select activation function
    def sel(self,args:dict):
        self.args = args
        select = args['pred_task']
        self.data_name = args['data_name']
        self.subset = args['subset']
        
        if(select == 'dp_hist'):
            self.dp_hist(args)
        if(select == 'dp_kde'):
            self.dp_kde(args)
        if(select == 'dp_bootstrap'):
            self.dp_bootstrap(args)
        if(select == 'dp_wildbootstrap'):
            self.dp_wildbootstrap(args)

    # for converting numeric text into int/float
    def convert_str(self,key:str):
        try:
            try:
                # if args[key] is a string
                val = eval(self.args[key])
            except:

                # else just a value
                val = self.args[key]
        except:
            val = None
        return val

    '''

    Activation Functions

    '''

    # plot Histogram of Two Samples (use plotly express)
    # which don't necessarily have the same sample size 

    def dp_hist(self,args:dict):

        sample1 = args['data'][0]
        sample2 = args['data'][1]

        data1 = pd.DataFrame(sample1,columns=['data'])
        data1['sample'] = 'one'
        data2 = pd.DataFrame(sample2,columns=['data'])
        data2['sample'] = 'two'
        names = ['one','two']
        combined = pd.concat([data1,data2])

        means_data = combined.groupby(by='sample').agg('mean')
        means = list(means_data['data'])

        fig = px.histogram(combined,x='data',color='sample',
                           marginal="box",
                           template='plotly_white',nbins=args['nbins'],
                           color_discrete_sequence=self.default_colors[0],
                           title='Comparing univariate distributions')

        fig.update_traces(opacity=0.8)
        fig.update_layout(barmode='group') # ['stack', 'group', 'overlay', 'relative']
        fig.update_layout(height=350,width=700)  
        # fig.update_traces(marker_line_width=1,marker_line_color="white") # outline
        fig.show()

    # plot Kernel Density Plot of Two Samples

    def dp_kde(self,args:dict):

        sample1 = args['data'][0]
        sample2 = args['data'][1]
        names = ['Sample 1','Sample 2']

        fig,ax = plt.subplots(1,1,figsize=(7,3.5))

        # Create a kernel density plot
        sns.kdeplot(data=[sample1, sample2],palette=self.default_colors[1],ax=ax,fill=True)
        sns.histplot(data=[sample1, sample2],palette=self.default_colors[1],ax=ax,alpha=0.01,stat='density',edgecolor=(0, 0, 0, 0.01))
        plt.legend(names)
        plt.xlabel('Values')
        plt.ylabel('Density')
        # plt.title('Distribution of Two Samples')
        plt.title('Distribution of Two Samples', loc='left', pad=10, fontdict={'horizontalalignment': 'left'})
        sns.despine(left=True)
        plt.tight_layout()
        plt.show()

    # plot Bootstrap Histogram Distribution 

    def dp_bootstrap(self,args:dict):

        pre = {'nsamples':100}

        sample1 = np.array(args['data'][0])
        sample2 = np.array(args['data'][1])

        # Number of bootstrap samples
        num_bootstrap_samples = sfpne(args,pre,'nsamples')

        # Perform bootstrap sampling and compute test statistic for each sample
        data = {'one':[],'two':[]}
        for i in range(num_bootstrap_samples):

            # Resample with replacement
            bootstrap_sample1 = np.random.choice(sample1, size=len(sample1), replace=True)
            bootstrap_sample2 = np.random.choice(sample2, size=len(sample2), replace=True)
            
            # Compute difference in CTR for bootstrap sample
            data['one'].append(np.mean(bootstrap_sample1))
            data['two'].append(np.mean(bootstrap_sample2))

        fig = px.histogram(data,x=['one','two'],
                           marginal="box",
                           template='plotly_white',nbins=args['nbins'],
                           color_discrete_sequence=self.default_colors[0],
                           title='Comparing Bootstrap distributions')

        fig.update_traces(opacity=0.8)
        fig.update_layout(barmode='group') # ['stack', 'group', 'overlay', 'relative']
        fig.update_layout(height=350,width=700)  
        fig.show()


    # plot Wild Bootstrap Histogram Distribution 

    # Wild Bootstrap: This method is useful when dealing with heteroscedastic data or data with dependence structures. 
    # It involves resampling the residuals from a model fitted to the original data, rather than resampling the original data itself. 
    # The A/B test is performed on each bootstrap sample of residuals, and the test statistic values are recorded.

    def dp_wildbootstrap(self,args:dict):

        pre = {'nsamples':100}

        sample1 = np.array(args['data'][0])
        sample2 = np.array(args['data'][1])

        # Number of bootstrap samples
        num_bootstrap_samples = sfpne(args,pre,'nsamples')
        
        # Function to estimate parameter
        def estimate_parameter(data):
            return np.mean(data)
        
        # Perform Wild Bootstrap
        boot1 = IIDBootstrap(sample1)
        boot_estimates1 = boot1.apply(estimate_parameter, num_bootstrap_samples)
        boot2 = IIDBootstrap(sample2)
        boot_estimates2 = boot2.apply(estimate_parameter, num_bootstrap_samples)
        
        data = {'one':boot_estimates1[:,0],'two':boot_estimates2[:,0]}

        fig = px.histogram(data,x=['one','two'],
                           marginal="box",
                           template='plotly_white',nbins=args['nbins'],
                           color_discrete_sequence=self.default_colors[0],
                           title='Comparing Wild Bootstrap distributions')

        fig.update_traces(opacity=0.8)
        fig.update_layout(barmode='group') # ['stack', 'group', 'overlay', 'relative']
        fig.update_layout(height=350,width=700)  
        fig.show()


