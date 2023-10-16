
### **Group**

eda


### **Purpose**

The purpose of this module is to 


### Selection 

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

### Activation Functions

Here you will find the relevant activation functions available in class `meda_scplot`

!!! note ""

	```python
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