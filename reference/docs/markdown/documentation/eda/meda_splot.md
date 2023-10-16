
## **Module Group**

`src/eda`[^1]

## **Project Stage ID**

[^1]: Reference to the sub folder in `src`

4[^2]

[^2]: Reference to the machine learning project phase identification defined [here](../../mlproject.md)

## :material-frequently-asked-questions: **Purpose**

The purpose of this module is to provide the user with the ability to utilise the basic visualisation tools provided in the library [seaborn](https://seaborn.pydata.org/)

## :fontawesome-solid-location-arrow: **Location**

Here are the locations of the relevant files associated with the module

### module information:

`/corpus/meda_splot.json`[^3]

[^3]: [location](../../../src/mllibs/corpus/meda_splot.json) | [github](https://github.com/shtrausslearning/mllibs/blob/main/src/mllibs/corpus/meda_splot.json)

### module activation functions

`/src/eda/meda_splot.py`[^4]

[^4]: [location](../../../src/mllibs/eda/meda_splot.py) | [github](https://github.com/shtrausslearning/mllibs/blob/main/src/mllibs/eda/meda_splot.py)

## :material-import: **Requirements**

Module import information

```python
from mllibs.nlpi import nlpi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json
```

## :material-selection-drag: **Selection**

Activation functions need to be assigned a unique label. Here's the process of `label` & activation function selection 

```python
    # called in nlpi

    def sel(self,args:dict):
                
        select = args['pred_task']
        self.data_name = args['data_name']
        
        ''' 
        
        ADD EXTRA COLUMNS TO DATA 

        model_prediction | splits_col

        
        '''
        # split columns (tts,kfold,skfold) 
        if(len(nlpi.data[self.data_name[0]]['splits_col']) != 0):

            split_dict = nlpi.data[self.data_name[0]]['splits_col']
            extra_columns = pd.concat(split_dict,axis=1)
            args['data'] = pd.concat([args['data'],extra_columns],axis=1)

        # model predictions
        if(len(nlpi.data[self.data_name[0]]['model_prediction']) != 0):

            prediction_dict = nlpi.data[self.data_name[0]]['model_prediction']
            extra_columns = pd.concat(prediction_dict,axis=1)
            extra_columns.columns = extra_columns.columns.map('_'.join)
            args['data'] = pd.concat([args['data'],extra_columns],axis=1)


        ''' 
        
        Activatation Function
        
        '''

        if(select == 'sscatterplot'):
            self.seaborn_scatterplot(args)
        elif(select =='srelplot'):
            self.seaborn_relplot(args)
        elif(select == 'sboxplot'):
            self.seaborn_boxplot(args)
        elif(select == 'sresidplot'):
            self.seaborn_residplot(args)
        elif(select == 'sviolinplot'):
            self.seaborn_violinplot(args)
        elif(select == 'shistplot'):
            self.seaborn_histplot(args)
        elif(select == 'skdeplot'):
            self.seaborn_kdeplot(args)
        elif(select == 'slmplot'):
            self.seaborn_lmplot(args)
        elif(select == 'spairplot'):
            self.seaborn_pairplot(args)
        elif(select == 'slineplot'):
            self.seaborn_lineplot(args)
        elif(select == 'scorrplot'):
            self.seaborn_heatmap(args)
```

## :octicons-code-16: **Activation Functions**

Here you will find the relevant activation functions available in class `meda_scplot`

### :octicons-file-code-16: `sscatterplot`

#### description:

Make a scatter plot using seaborn, bivariate or multivariate

#### code:

```python linenums="1"
@staticmethod
def seaborn_scatterplot(args:dict):
       
    if(args['hue'] is not None):

        hueloc = args['data'][args['hue']]
        if(type(nlpi.pp['stheme']) is str):
            palette = nlpi.pp['stheme']
        else:
            palette = palette_rgb[:len(hueloc.value_counts())]
            
    else:
        hueloc = None
        palette = palette_rgb
        
    sns.set_style("whitegrid", {
        "ytick.major.size": 0.1,
        "ytick.minor.size": 0.05,
        'grid.linestyle': '--'
    })

    sns.scatterplot(x=args['x'], 
                    y=args['y'],
                    hue=args['hue'],
                    alpha = nlpi.pp['alpha'],
                    linewidth=nlpi.pp['mew'],
                    edgecolor=nlpi.pp['mec'],
                    s = nlpi.pp['s'],
                    data=args['data'],
                    palette=palette)
    
    sns.despine(left=True, bottom=True)
    plt.show()
    nlpi.resetpp()
```

### :octicons-file-code-16: `srelplot`

#### description:

Make a scatter plot using seaborn, bivariate or multivariate

#### code:

```python linenums="1"
@staticmethod
def seaborn_relplot(args:dict):
        
    if(args['hue'] is not None):
        hueloc = args['data'][args['hue']]
        if(type(nlpi.pp['stheme']) is str):
            palette = nlpi.pp['stheme']
        else:
            palette = palette_rgb[:len(hueloc.value_counts())]
            
    else:
        hueloc = None
        palette = palette_rgb           
        
    sns.set_style("whitegrid", {
        "ytick.major.size": 0.1,
        "ytick.minor.size": 0.05,
        'grid.linestyle': '--'
    })
    
    sns.relplot(x=args['x'], 
                y=args['y'],
                col=args['col'],
                row=args['row'],
                hue=args['hue'], 
                col_wrap=args['col_wrap'],
                kind=args['kind'],
                palette=palette,
                alpha= nlpi.pp['alpha'],
                s = nlpi.pp['s'],
                linewidth=nlpi.pp['mew'],
                edgecolor=nlpi.pp['mec'],
                data=args['data'])
    
    sns.despine(left=True, bottom=True)
    plt.show()
    nlpi.resetpp()
```
### :octicons-file-code-16: `sboxplot`

#### description:

Create a seaborn box plot using boxpot, they show quartiles and outliers

#### code:

```python linenums="1"
@staticmethod
def seaborn_boxplot(args:dict):
    
    if(args['hue'] is not None):
        hueloc = args['data'][args['hue']]
        if(type(nlpi.pp['stheme']) is str):
            palette = nlpi.pp['stheme']
        else:
            palette = palette_rgb[:len(hueloc.value_counts())]
            
    else:
        hueloc = None
        palette = palette_rgb
        
    sns.set_style("whitegrid", {
        "ytick.major.size": 0.1,
        "ytick.minor.size": 0.05,
        'grid.linestyle': '--'
    })
    
    if(args['bw'] is None):
        bw = 0.8
    else:
        bw = eval(args['bw'])
    
    sns.boxplot(x=args['x'], 
                y=args['y'],
                hue=args['hue'],
                width=bw,
                palette=palette,
                data=args['data'])
    
    sns.despine(left=True, bottom=True)
    plt.show()
```
