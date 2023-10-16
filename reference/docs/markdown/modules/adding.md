## Overview

To add a new module into `mllibs`, you need to add two components:

- A module `configuration` file, located in `/src/mllibs/corpus/` 
- A module `components` file, located in `src/mllibs/`

The `configuration` file will hold text based information about each activation function in the module, whilst the `components` file will contain the relevant activation functions


## Module Components File

Module activation functions are grouped together in a class format, here is an example module, `sample`, which contains 

```python
# sample module class structure
class sample(nlpi):
    
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
        
    @staticmethod
    def function(args:dict):
        pass
```

## Module Configuration File

The congiguration file contains information about the module & its stored functions `info`, as well as the `corpus` used in classificaiton of function labels `name`

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
