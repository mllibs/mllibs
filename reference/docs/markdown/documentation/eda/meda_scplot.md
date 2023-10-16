
## **Module Group**

`eda`

## **Project Stage ID**

4

## **Purpose**

The purpose of this module is to provide the user with the ability to visualise each numerical columns in a pandas dataframe in a two dimensional figure relative to other numerical columns


## Selection 

The process of `label` & activation function selection

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

## Activation Functions

Here you will find the relevant activation functions available in class `meda_scplot`

### :octicons-file-code-16: `col_kde`

!!! info "col_kde"

	```python title="col_kde" linenums="1" hl_lines="35 45"
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

!!! info "col_box"

	```python title="col_box" linenums="1" hl_lines="47 54"
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
