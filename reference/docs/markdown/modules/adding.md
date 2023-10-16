## Overview

Adding a new module into `mllibs` requires two components:

- A configuration file, located in `/src/mllibs/corpus/` 
- A module function class, located in `src/mllibs/`

## Module Components File

```python
# sample module class structure
class sample(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        self.name = 'sample'             # unique module name identifier (used in nlpm/nlpi)
        self.nlp_config = nlp_config  # text based info related to module (used in nlpm/nlpi)
        
    # called in nlpi
    def sel(self,args:dict):
        
        self.select = args['pred_task']
        self.args = args
        
        if(self.select == 'function'):
            self.function(self.args)
        
    # use standard or static methods
        
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
  "name": "col_kde",
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
