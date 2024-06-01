
from mllibs.nlpi import nlpi
from mllibs.dict_helper import sfp,sfpne, convert_str_to_val, column_to_subset
from mllibs.df_helper import check_list_in_col
import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json
from scipy.fft import fft, fftfreq
import seaborn as sns
import matplotlib.pyplot as plt

'''

Fourier Transformation Related

'''

class fourier_all(nlpi):
    
    def __init__(self):
        self.name = 'fourier_all'  

        path = pkg_resources.resource_filename('mllibs','/signal/mfourier_all.json')
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
        select = args['pred_task']

        if(select == 'sig_fourier'):
            self.sig_fourier(args)
        if(select == 'sig_fouriers'):
            self.sig_fouriers(args)
        if(select == 'sig_fourierplot'):
            self.sig_fourierplot(args)

    '''

    Calculate FFT for pandas Series

    '''

    @staticmethod
    def get_fft(data:pd.Series):

        # numpy
        data = data.values
        fft_result = fft(data) # Calculate the FFT using scipy
        
        # Calculate the frequency values corresponding to the FFT coefficients in Hz
        n = len(data)  # Length of the input data
        timestep = 1  # Assume a unit timestep for simplicity
        freq = fftfreq(n, d=timestep)
        freq_in_hz = np.fft.fftshift(freq) * n  # Convert normalized frequency to Hz
        
        # Calculate the magnitude of the FFT coefficients
        magnitude = np.abs(fft_result)
        
        df = pd.DataFrame({'freq':freq_in_hz,
                           'magnitude':magnitude})
        df = df[df['freq'] > 0]
        return df

    '''
    ####################################################################################

    Activation Functions

    ####################################################################################
    '''

    # Fourier transformation for a single column
    # requires column information from dataframe ie. check

    def sig_fourier(self,args:dict):

        # subset : (str/None)
        subset = column_to_subset(args)

        if(subset != None):
            fourier_data = self.get_fft(args['data'][args['column']])
            nlpi.memory_output.append({'data':fourier_data})
        else:
            print('[note] please reference a column you want to FFT')

    # Fourier transformation for multiple columns
    # requires column information from dataframe ie. check
            
    # args['column'] : multiple column names

    def sig_fouriers(self,args:dict):

        # subset : (str/None)
        subset = column_to_subset(args)

        if(subset != None):

            lst_data = []
            for column in args['column']:
                fourier_data = self.get_fft(args['data'][column])
                lst_data.append(fourier_data)

            df_fft = pd.concat(lst_data,axis=1)
            nlpi.memory_output.append({'data':df_fft})

        else:
            print('[note] please reference the columns you want to FFT')

    # Fourier transformation and plot

    def sig_fourierplot(self,args:dict):
        
        # subset treatment 

        def check_column_format(lst:list):
            
            if(len(lst) == 1):
                return [lst[0]]

            # more than one column is found 
            elif(len(lst) == 2):
                if(nlpi.silent == False):
                    print("[note] I'll group the specified columns together")
                
                # nested lists to single list
                # taking into account str entries
                grouped = []
                for item in lst:
                    if isinstance(item, list):
                        grouped.extend(item)
                    elif isinstance(item, str):
                        grouped.append(item)
                    else:
                        grouped.append(item)

                return grouped 
            
            else:
                print('[note] please specify the columns you want to transform only')
                return None

        def plot_cols(df:pd.DataFrame,cols:list):
            
            # check correctness of subset column format
            cols = check_column_format(cols)

            # check if column is in dataframe
            if(cols != None):
                col_check = check_list_in_col(df,cols)
            else:
                col_check = False

            if(col_check):
                
                lst_df = []
                for col in cols:
                    df_fft = self.get_fft(df[col])
                    df_fft['case'] = df[col].name
                    lst_df.append(df_fft)
            
                if(len(lst_df) > 1):
                    merged_df = pd.concat(lst_df, ignore_index=True)
                else:
                    merged_df = lst_df[0]

                def seaborn_setstyle():
                    sns.set_style("whitegrid", {
                      "ytick.major.size": 0.1,
                      "ytick.minor.size": 0.05,
                      'grid.linestyle': '--'
                    })

                if(nlpi.pp['figsize'] == None):
                    fsize = (10,5)
                else:
                    fsize = nlpi.pp['figsize']

                seaborn_setstyle()

                fig, ax = plt.subplots(figsize=fsize)
                groups = dict(tuple(merged_df.groupby('case')))

                for group,gdata in groups.items():   
                    sns.lineplot(gdata,x='freq',y='magnitude',label=group,ax=ax)

                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
                ax.legend()
                plt.tight_layout()
                sns.despine(left=True,right=True,top=True,bottom=True)
                plt.show()

            else:
                print('[note] column not present in dataframe')

        # subset 
        subset = column_to_subset(args)

        if(subset != None):
            plot_cols(args['data'],subset)