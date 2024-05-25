
from mllibs.nlpi import nlpi
from mllibs.dict_helper import sfp,sfpne
from mllibs.module_helper import confim_dtype, get_ddata, get_mdata, get_sdata, get_dmdata, get_spdata
import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json
import seaborn as sns
import matplotlib.pyplot as plt
from mllibs.module_helper import confim_dtype
import textwrap

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
        default_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'] # express new
        # default_colors = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477','#66AA00','#B82E2E','#316395'] # g10
        self.default_colors = default_colors
        
    def sel(self,args:dict):

        '''

        Relevant Activation function is selected 

        '''

        self.args = args
        select = args['pred_task']
        self.data_name = args['data_name']
        self.subset = args['subset']
        self.info = args['task_info']['description']

        def get_data(dtype_id:str,case_id:str,names_id=False):

            '''

            Get the data (names_id == False) or just return names if (names_id == True)

            '''

            if(case_id == 'sdata'):
                ldata_names = get_sdata(args['data'],dtype_id)
            elif(case_id == 'ddata'):
                ldata_names = get_ddata(args['data'],dtype_id)
            elif(case_id == 'mdata'):
                ldata_names = get_mdata(args['data'],dtype_id)
            elif(case_id == 'dmdata'):
                ldata_names = get_dmdata(args['data'],dtype_id)
            elif(case_id == 'spdata'):
                ldata_names = get_spdata(args['data'],dtype_id)

            # return data or names only
            if(names_id == False):
                if(ldata_names != None):
                    if(len(ldata_names) != 1):
                        ldata = []
                        for dname in ldata_names:
                            ldata.append(nlpi.data[dname]['data'])
                        return ldata
                    elif(len(ldata_names) == 1):
                        return nlpi.data[ldata_names[0]]['data']
                else:
                    return None 
            else:
                if(ldata_names != None):
                    return ldata_names
                else:
                    return None
    
        if(nlpi.silent == False):
            print('\n[note] module function info');print(textwrap.fill(self.info, 60));print('')

        '''
        ////////////////////////////////////////////////////////

                        Select Activation Function

        ////////////////////////////////////////////////////////
        '''  

        if(select == 'dp_hist'):

            args['data'] = get_data('list','spdata',True)
            if(args['data'] != None):
                self.dp_hist(args)
        
        if(select == 'dp_kde'):

            args['data'] = get_data('list','spdata',True)
            if(args['data'] != None):
                self.dp_kde(args)

        if(select == 'dp_box'):

            args['data'] = get_data('list','spdata',True)
            if(args['data'] != None):
                self.dp_box(args)
        
        if(select == 'dp_ecdf'):

            args['data'] = get_data('list','spdata',True)
            if(args['data'] != None):
                self.dp_ecdf(args)


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
    ////////////////////////////////////////////////////////////

                       Activation Functions

    ////////////////////////////////////////////////////////////
    '''

    def dp_hist(self,args:dict):

        '''

        Plot the Histrogram Univariate Distribution

        '''

        # [when lists] combine all dictionaries into one dataframe

        lst_ldata = []
        ldata = self.args['data']
        for key in ldata:
            ldata = pd.DataFrame(nlpi.data[key]['data'],columns=['data'])
            ldata['sample'] = key
            lst_ldata.append(ldata)

        combined = pd.concat(lst_ldata)
        combined = combined.reset_index(drop=True)

        if(args['nbins'] is not None):
            bins = args['nbins']
        else:
            print('[note] default [nbins] parameter set')
            bins = 100 

        # plot the histograms

        fig, ax = plt.subplots(figsize=(10,4))
        sns.despine(left=True,right=True,top=True,bottom=True)
        plt.grid(linestyle='--', linewidth=0.5,alpha=0.2)
        
        sns.histplot(combined,x='data',hue='sample',bins=bins,
                     alpha=0.5,edgecolor='k',linewidth=1,
                     palette=self.default_colors,ax=ax)

        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Univariate Distribution')

    def dp_kde(self,args:dict):

        '''

        Plot the Kernel Density Univariate Distribution

        '''

        # [when lists] combine all dictionaries into one dataframe

        lst_ldata = []
        ldata = self.args['data']
        for key in ldata:
            ldata = pd.DataFrame(nlpi.data[key]['data'],columns=['data'])
            ldata['sample'] = key
            lst_ldata.append(ldata)

        combined = pd.concat(lst_ldata)
        combined = combined.reset_index(drop=True)

        # plot the kernel density plot

        fig, ax = plt.subplots(figsize=(10,4))
        sns.despine(left=True,right=True,top=True,bottom=True)
        plt.grid(linestyle='--', linewidth=0.5,alpha=0.2)

        # Create a kernel density plot
        sns.kdeplot(combined,x='data',hue='sample',palette=self.default_colors,ax=ax,fill=True)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Univariate Distribution')
        plt.show()


    def dp_box(self,args:dict):

        '''

        Plot the Boxplot Univariate Distribution

        '''

        # [when lists] combine all dictionaries into one dataframe

        lst_ldata = []
        ldata = self.args['data']
        for key in ldata:
            ldata = pd.DataFrame(nlpi.data[key]['data'],columns=['data'])
            ldata['sample'] = key
            lst_ldata.append(ldata)

        combined = pd.concat(lst_ldata)
        combined = combined.reset_index(drop=True)

        # plot boxplot using seaborn

        fig, ax = plt.subplots(figsize=(10,4))
        sns.despine(left=True,right=True,top=True,bottom=True)
        plt.grid(linestyle='--', linewidth=0.5,alpha=0.2)
        
        sns.boxplot(combined,x='sample',y='data',width=0.5,palette=self.default_colors,ax=ax)

        ax.set_xlabel('Subset')
        ax.set_ylabel('Value')
        ax.set_title('Univariate Distribution')
        plt.show()


    def dp_ecdf(self,args:dict):

        '''

        Plot the Univariate Cumulative Distribution Function

        '''

        # [when lists] combine all dictionaries into one dataframe

        lst_ldata = []
        ldata = self.args['data']
        for key in ldata:
            ldata = pd.DataFrame(nlpi.data[key]['data'],columns=['data'])
            ldata['sample'] = key
            lst_ldata.append(ldata)

        combined = pd.concat(lst_ldata)
        combined = combined.reset_index(drop=True)

        # plot boxplot using seaborn

        fig, ax = plt.subplots(figsize=(10,4))
        sns.despine(left=True,right=True,top=True,bottom=True)
        plt.grid(linestyle='--', linewidth=0.5,alpha=0.2)
        
        sns.ecdfplot(combined,x='data',hue='sample',palette=self.default_colors,ax=ax)

        ax.set_xlabel('data')
        ax.set_ylabel('proportion')
        ax.set_title('Cumulative Distribution Function')
        plt.show()


    # plot Bootstrap Histogram Distribution 

    def dp_bootstrap(self,args:dict):

        '''

        Take n-samples from 

        '''

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

        # fig = px.histogram(data,x=['one','two'],
        #                    marginal="box",
        #                    template='plotly_white',nbins=args['nbins'],
        #                    color_discrete_sequence=self.default_colors[0],
        #                    title='Comparing Bootstrap distributions')

        # fig.update_traces(opacity=0.8)
        # fig.update_layout(barmode='group') # ['stack', 'group', 'overlay', 'relative']
        # fig.update_layout(height=350,width=700)  
        # fig.show()
