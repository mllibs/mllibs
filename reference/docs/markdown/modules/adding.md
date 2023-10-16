## :octicons-multi-select-16: Overview

To add a new module into `mllibs`, you need to add two components:

- A module `configuration` file, located in `/src/mllibs/corpus/` 
- A module `components` file, located in `src/mllibs/`

The `configuration` file will hold text based information about each activation function in the module, whilst the `components` file will contain the relevant activation functions

## :material-inbox-multiple-outline: Module Components File

Module activation functions are grouped together in a class format. Here is an example module, `sample`, which contains an activation function `function`

### class inheritance

Modules can inherent any class, however as a minimum, it must always inherent the `nlpi` class

### activation functions

Activation functions require only a single argument, `args:dict` aside from `self`

```python
# sample module class structure

class Sample(nlpi):
    
    '''
	
	Initialise Module

    '''
    # called in nlpm

    def __init__(self,nlp_config):
        self.name = 'sample'             # unique module name identifier (used in nlpm/nlpi)
        self.nlp_config = nlp_config  # text based info related to module (used in nlpm/nlpi)
        
    '''

	Function Selector 

    '''
    # called in nlpi

    def sel(self,args:dict):
        
        self.select = args['pred_task']
        self.args = args
        
        if(self.select == 'function'):
            self.function(self.args)
        
    '''

	Activation Functions

    '''
        
    def function(self,args:dict):
        pass
```


## :octicons-git-merge-16: Module Configuration File

The `configuration` file contains information about the module (eg.`sample`) & its stored functions `info`, as well as the `corpus` used in classificaiton of function labels `name`

``` json
"modules": [

{
  "name": "function",
"corpus": [
          "...",
          ],
  "info": {
          "module":"sample",
          "action":"...",
          "topic":"...",
          "subtopic":"...",
          "input_format":"...",
          "description":"...",
          "output":"...",
          "token_compat":"...",
          "arg_compat":"..."
          }
},

...

]
```

## :material-checkbox-multiple-marked-circle-outline: Naming Conventions

### Activation function name

Some important things to note:

- Module class name (eg.`Sample`) can be whatever you choose. The relevant class must then be used as import when grouping together all other modules. 
- Module `configuration` must contain `name` (function names) that correspond to its relevant module 

### File names

Module `components` file names can be whatever you choose it to be. Module `configuration` file names as well can be anything you choose it to be, however its good practice to choose the same name for both module components so you don't loose track of which files belong together.
