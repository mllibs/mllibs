
## **Module Group**

`src/eda`[^1]

## **Project Stage ID**

[^1]: Reference to the sub folder in `src`

4[^2]

[^2]: Reference to the machine learning project phase identification defined [here](../../mlproject.md)

## :material-frequently-asked-questions: **Purpose**

The purpose of this module is to provide the user with the ability to visualise each numerical columns in a pandas dataframe in a two dimensional figure relative to other numerical columns, the module revolves around the utilisation of [seaborn](https://seaborn.pydata.org/)

## :fontawesome-solid-location-arrow: **Location**

Here are the locations of the relevant files associated with the module

<h4>module information:</h4>

`/corpus/meda_scplot.json`[^3]

[^3]: [location](../../../src/mllibs/corpus/meda_scplot.json) | [github](https://github.com/shtrausslearning/mllibs/blob/main/src/mllibs/corpus/meda_scplot.json)

<h4>module activation functions:</h4>

`/src/eda/meda_scplot.py`[^4]

[^4]: [location](../../../src/mllibs/eda/meda_scplot.py) | [github](https://github.com/shtrausslearning/mllibs/blob/main/src/mllibs/eda/meda_scplot.py)

## :material-import: **Requirements**

Module import information

```python
from mllibs.nlpi import nlpi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json
```

## :material-selection-drag: **Selection**

Activation functions need to be assigned a unique label. Here's the process of `label` & activation function selection 

```python
def sel(self,args:dict):
    
    select = args['pred_task']
              
    if(select == 'col_kde'):
        self.eda_colplot_kde(args)
    elif(select == 'col_box'):
        self.eda_colplot_box(args)
    elif(select == 'col_scatter'):
        self.eda_colplot_scatter(args)
```

## :octicons-code-16: **Activation Functions**

Here you will find the relevant activation functions available in class `meda_scplot`

### :octicons-file-code-16: `col_kde`

#### description:

visualise/plot column feature kernel density estimation plot

#### code:

```python linenums="1" hl_lines="35-45"
def eda_colplot_kde(self,args:dict):
    
    # get numeric column names only
    num,_ = self.split_types(args['data'])
        
    if(args['x'] is not None):
        xloc = args['data'][args['x']]
    else:
        xloc = None
        
    if(args['hue'] is not None):
        hueloc = args['data'][args['hue']]
        if(type(nlpi.pp['stheme']) is str):
            palette = nlpi.pp['stheme']
        else:
            palette = palette_rgb[:len(hueloc.value_counts())]
            
    else:
        hueloc = None
        palette = palette_rgb
      
    columns = list(num.columns)  
    n_cols = 3
    n_rows = math.ceil(len(columns)/n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
    ax = ax.flatten()

    for i, column in enumerate(columns):
        plot_axes = [ax[i]]
        
        sns.set_style("whitegrid", {
        'grid.linestyle': '--'})

        sns.kdeplot(data=args['data'],
                    x=column,
                    hue=hueloc,
                    fill=nlpi.pp['fill'],
                    alpha= nlpi.pp['alpha'],
                    linewidth=nlpi.pp['mew'],
                    edgecolor=nlpi.pp['mec'],
                    ax=ax[i],
                    common_norm=False,
                    palette=palette
                     )

        # titles
        ax[i].set_title(f'{column} distribution');
        ax[i].set_xlabel(None)

    for i in range(i+1, len(ax)):
        ax[i].axis('off')
                  
    plt.tight_layout()
```

### :octicons-file-code-16: `col_box`

#### description:

visualise/plot column feature boxplot

#### code:

```python linenums="1" hl_lines="47-54"
def eda_colplot_box(self,args:dict):

# split data into numeric & non numeric
num,cat = self.split_types(args['data'])
  
columns = list(num.columns)  
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)

if(args['x'] is not None):
    xloc = args['data'][args['x']]
else:
    xloc = None
    
if(args['x'] is not None):
    xloc = args['data'][args['x']]
else:
    xloc = None
    
if(args['hue'] is not None):
    hueloc = args['data'][args['hue']]
    if(type(nlpi.pp['stheme']) is str):
        palette = nlpi.pp['stheme']
    else:
        palette = palette_rgb[:len(hueloc.value_counts())]
        
else:
    hueloc = None
    palette = palette_rgb

fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
sns.despine(fig, left=True, bottom=True)
ax = ax.flatten()

for i, column in enumerate(columns):
    plot_axes = [ax[i]]
    
    sns.set_style("whitegrid", {
    'grid.linestyle': '--'})


    if(args['bw'] is None):
        bw = 0.8
    else:
        bw = eval(args['bw'])

    sns.boxplot(
        y=args['data'][column],
        x=xloc,
        hue=hueloc,
        width=bw,
        ax=ax[i],
        palette=palette
    )

    # titles
    ax[i].set_title(f'{column} distribution');
    ax[i].set_xlabel(None)
    
    
for i in range(i+1, len(ax)):
    ax[i].axis('off')

plt.tight_layout()
```

***

### :octicons-file-code-16: `col_scatter`

#### description:

visualise/plot column feature scatterplot

#### code:

```python linenums="1" hl_lines="36-46"
    def eda_colplot_scatter(self,args:dict):

        # split data into numeric & non numeric
        num,_ = self.split_types(args['data'])
          
        columns = list(num.columns)  
        n_cols = 3
        n_rows = math.ceil(len(columns)/n_cols)
        
        if(args['x'] is not None):
            xloc = args['data'][args['x']]
        else:
            xloc = None
            
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
        sns.despine(fig, left=True, bottom=True)
        ax = ax.flatten()

        for i, column in enumerate(columns):
            plot_axes = [ax[i]]
            
            sns.set_style("whitegrid", {
            'grid.linestyle': '--'})

            sns.scatterplot(
                y=args['data'][column],
                x=xloc,
                hue=hueloc,
                alpha= nlpi.pp['alpha'],
                linewidth=nlpi.pp['mew'],
                edgecolor=nlpi.pp['mec'],
                s = nlpi.pp['s'],
                ax=ax[i],
                palette=palette,
            )

            # titles
            ax[i].set_title(f'{column} distribution');
            ax[i].set_xlabel(None)
            
            
        for i in range(i+1, len(ax)):
            ax[i].axis('off')
        
        plt.tight_layout()
        plt.show()
```

