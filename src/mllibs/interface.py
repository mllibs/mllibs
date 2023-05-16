# NLP module
from mllibs.nlpi import nlpi
from mllibs.nlpm import nlpm

# import additional modules
from mllibs.mloader import loader,configure_loader
from mllibs.mseda import simple_eda,configure_eda
from mllibs.meda_splot import eda_plot, configure_edaplt
from mllibs.meda_scplot import eda_colplot, configure_colplot
from mllibs.mpd_df import dataframe_oper, configure_pda
from mllibs.mnlp_encoder import encoder, configure_nlpencoder


# single command interpreter, multiple command interpreter & interface (chat)


'''

Single command interpreter interface

'''

class snlpi(nlpi):
    
    def __init__(self,collection):
        super().__init__(collection)
        
    def exec(self,command:str,args:dict=None):  
        self.do(command=command,args=args)
            

'''

Multiple command interpreter interface

'''
    
class mnlpi(nlpi):
    
    def __init__(self,collection):
        super().__init__(collection)
        
    def exec(self,command:str,args:dict=None):  
        
        # criteria for splitting (just test)
        strings = command.split(',')    
        
        for string in strings:
            self.do(command=string,args=args)
            
            

'''

Main user interfacing class 

'''

# interface class is a user interaction class

class interface(snlpi,mnlpi,nlpi):

    def __init__(self):
        
        # compile modules
        self.collection = self.prestart()
        snlpi.__init__(self,self.collection) 
               

    def __getitem__(self,command:str):
        self.exec(command,args=None)
        

    def prestart(self):

        collection = nlpm()
        collection.load([loader(configure_loader),        # load data
                         simple_eda(configure_eda),       # pandas dataframe information
                         eda_plot(configure_edaplt),      # standard visuals
                         eda_colplot(configure_colplot),  # column based visuals
                         dataframe_oper(configure_pda),   # pandas dataframe operations
                         encoder(configure_nlpencoder),
                        # embedding(configure_nlpembed),
                        # cleantext(configure_nlptxtclean),
                        # sklinear(configure_sklinear),
                        # hf_pipeline(configure_hfpipe),
                        # data_outliers(configure_outliers),
                        ])


        collection.train()
                            
        return collection
        
    
    def iter_loop(self):
        
        # user command 
        if(command == None):
            print('What would you like to do?')
            self.command = input()
        else:
            self.command = command
            
        ''' Check for multicommand '''
        # currently simple implementation based on rules
        
        tokens = nlpi.nltk_tokeniser(self.command)
        
        for token in tokens:
            if(token in text_store.dividers):
                ctype = 'multiple'
            else:
                ctype = 'single'
        
        # activate relevant interpreter
        if(ctype == 'multiple'):
            mnpli.__init__(self,self.collection)
            self.exec(str(self.command))
        elif(ctype == 'single'):
            snlpi.__init__(self,self.collection)
            self.exec(str(self.command))
            self.return_data()
            
            
            
    def return_data(self):
        print('storing data in global variable: stored')
        globals()['stored'] = self.glr()
        
        