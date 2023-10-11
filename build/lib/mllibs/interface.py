# NLP module
from mllibs.nlpi import nlpi
from mllibs.nlpm import nlpm

# import additional modules
from mllibs.mloader import loader,configure_loader
from mllibs.mseda import simple_eda,configure_eda
from mllibs.meda_splot import eda_plot, configure_edaplt
from mllibs.meda_scplot import eda_colplot, configure_colplot


class snlpi(nlpi):
    
    def __init__(self,collection):
        super().__init__(collection)
        
    def exec(self,command:str,args:dict=None):  
        self.do(command=command,args=args)
            
    
class mnlpi(nlpi):
    
    def __init__(self,collection):
        super().__init__(collection)
        
    def exec(self,command:str,args:dict=None):  
        
        # criteria for splitting (just test)
        strings = command.split(',')    
        
        for string in strings:
            self.do(command=string,args=args)
        

class interface(snlpi,mnlpi,nlpi):

    def __init__(self,silent=False):
        
        # compile modules
        self.collection = self.prestart()
        snlpi.__init__(self,self.collection)
        if(silent is False):
            nlpi.silent = False
        else:
            nlpi.silent = True 
               
    def __getitem__(self,command:str):
        self.exec(command,args=None)
        

    def prestart(self):

        collection = nlpm()
        collection.load([loader(configure_loader),        # load data
                         simple_eda(configure_eda),       # pandas dataframe information
                         eda_plot(configure_edaplt),      # standard visuals
                         eda_colplot(configure_colplot),  # column based visuals
                        ])


        collection.train()
                            
        return collection
        
        
        