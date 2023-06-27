# importing the dependencies needed for pre processing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import emoji
import string
from mllibs.nlpi import nlpi
import pandas as pd
import spacy


class cleantext(nlpi):
    
    def __init__(self,nlp_config):
        self.name = 'nlp_cleantext'
        self.nlp_config = nlp_config 
        self.select = None
        self.data = None
        self.args = None
           
    # make selection  

    def sel(self,args:dict):
    
        self.select = args['pred_task']
        self.data = args['data']
        self.args = args  
        
        ''' select appropriate predicted method '''
        
        if(self.select == 'clean_text'):
            self.clean_text_general(self.data,self.args)
        elif(self.select == 'lemma_text'):
            self.lemmatise(self.data,self.args)
        elif(self.select == 'norm_text'):
            self.normalise_text(self.data,self.args)
        elif(self.select == 'convert_emoji'):
            self.convert_emoji(self.data,self.args)
        elif(self.select == 'remove_emoji'):
            self.remove_emoji(self.data,self.args)
        elif(self.select == 'remove_special_char'):
            self.remove_special_char(self.data,self.args)
        elif(self.select == 'remove_http'):
            self.remove_http(self.data,self.args)
        elif(self.select == 'remove_numbers'):
            self.remove_numbers(self.data,self.args)
        elif(self.select == 'remove_whitespace'):
            self.remove_whitespace(self.data,self.args)
        elif(self.select == 'stemmer'):
            self.stemmer(self.data,self.args)
        elif(self.select == 'remove_handle'):
            self.remove_handle(self.data,self.args)
            
    def clean_text_general(self,data:pd.Series,args):
        
        en_stopwords = stopwords.words("english") # stop words 
        lemma = WordNetLemmatizer() # lemmatiser

        # define a function for preprocessing
        def text_cleaning(ltext):
#             print(type(ltext))
            # ltext = ltext.replace(np.nan, '')
            ltext = re.sub("[^A-Za-z1-9 ]", "", ltext) #removes punctuation marks
            ltext = ltext.lower() #changes to lower case
            tokens = word_tokenize(ltext) #tokenize the text
            clean_list = [] 
            for token in tokens:
                if token not in en_stopwords: #removes stopwords
                    clean_list.append(lemma.lemmatize(token)) #lemmatizing and appends to clean_list
            return " ".join(clean_list)# joins the tokens
          
        nlpi.memory_output.append(data.apply(text_cleaning) )
        
        
    '''
    
    Text Lemmatisation
    
    
    '''
        
    def lemmatise(self,data:pd.Series,args):

        # define a function for preprocessing
        def clean(text):
            nlp = spacy.load("en_core_web_sm")
            
            lst_cleaned = [] 
            for doc in nlp.pipe(text, 
                                batch_size=32, 
                                n_process=3, 
                                disable=["parser", "ner"]):
                                    
                lst_lemma = [tok.lemma_ for tok in doc]
                lst_cleaned.append(" ".join(lst_lemma))
            
            return lst_cleaned
    
        lst_lem = clean(data)
        nlpi.memory_output.append(pd.Series(lst_lem))
        
    def normalise_text(self,data:pd.Series,args):
        
        def normalise_tokens(ltext):
            ltext = ltext.lower() #changes to lower case
            tokens = word_tokenize(ltext) #tokenize the text
            token_list = [] 
            for token in tokens:
                token_list.append(token) 
            return " ".join(token_list)# joins the tokens
           
        nlpi.memory_output.append(data.apply(normalise_tokens))
        
    def convert_emoji(self,data:pd.Series,args):   
        def remove(text):
            return emoji.demojize(text, language='en')

        nlpi.memory_output.append(data.apply(remove))

    def remove_emoji(self,data:pd.Series,args):   
        
        def remove_converted_emoji(ltext):
            ltext = emoji.demojize(ltext, language='en')
            ltext = re.sub(r':.+?:', '', ltext)
            tokens = word_tokenize(ltext) #tokenize the text
            token_list = [] 
            for token in tokens:
                token_list.append(token) 
            return " ".join(token_list)# joins the tokens
        
        nlpi.memory_output.append(data.apply(remove_converted_emoji))
    
    # remove punctuation
    
    def remove_special_char(self,data:pd.Series,args):    
        
        punctuations_list = string.punctuation 
        
        def clean_punctuation(text):
            transformator = str.maketrans('', '', punctuations_list)
            return text.translate(transformator)
        
        nlpi.memory_output.append(data.apply(clean_punctuation))
     
    # remove links
    
    def remove_http(self,data:pd.Series,args):

        def removelink(text):
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"\w+@\w+\.com", "", text)
            tokens = word_tokenize(text) #tokenize the text
            token_list = [] 
            for token in tokens:
                token_list.append(token) 
            return " ".join(token_list)# joins the tokens
              
        nlpi.memory_output.append(data.apply(removelink))
        
    # remove numbers
    
    def remove_numbers(self,data:pd.Series,args):
        
        def clean_num(text):       
            return re.sub(r"[0-9]+", "", text)
    
        nlpi.memory_output.append(data.apply(clean_num))    
        
    # remove whitespaces
    
    def remove_whitespace(self,data:pd.Series,args):
        
        def clean_wspace(text):
            text = text.strip()  # Leading and trailing whitespaces are removed
            return re.sub(r" +"," ",text)
        
        nlpi.memory_output.append(data.apply(clean_wspace))    
     
    # stemmer simplification/normalisation
        
    def stemmer(self,data:pd.Series,args):
        
        # Creating an instance of the stemmer :
        stemmer = SnowballStemmer(language='english')

        # Creating a fucntion that will be applied to our datset :
        def clean_stem(text):
            tokens = word_tokenize(text) #tokenize the text
            token_list = [] 
            for token in tokens:        
                token_list.append(stemmer.stem(token)) 
            return " ".join(token_list)

        ## Applying the fucntion to all rows :
        nlpi.memory_output.append(data.apply(clean_stem))
        
    # remove twitter handle
        
    def remove_handle(self,data:pd.Series,args):
        
        def clean_mention(text):
            ltext = re.sub(r"@\S+", "", text)
            return ltext
        
        nlpi.memory_output.append(data.apply(clean_mention))
    
        
    
    
        
    
    
dict_nlptxtclean = {'clean_text':['clean text',
                                   'general text cleaning',
                                   'common text simplification',
                                   'cleaning of text',
                                   'simplify text',
                                   'tidy text',
                                   'remove unwanted text',
                                   'text cleaning'],
                     
                     'lemma_text':['lemmatise text',
                                      'lemmatisation of text',
                                      'create base form',
                                      'base form normalisation',
                                      'lemma text',
                                      'lemmatise'],
                                   
                    'norm_text':['normalise text',
                                 'lower register',
                                 'normalise text',
                                 'lower text register'],
                    
                    'convert_emoji':['convert emoji to text',
                                     'convert emoji',
                                    'translate emoji icons',
                                    'translate emoji'
                                    'emoji translate'],
                    
                    'remove_emoji':['remove emoji',
                                   'get rid of emoji',
                                   'remove emoji icons',
                                   'clean emoji icons',
                                   'emoji normalisation',
                                   'clean emoji'],
                    
                    'remove_special_char': ['remove special characters',
                                           'clean special characters',
                                           'remove punctuation',
                                           'remove brackets',
                                           'get rid of punctuation',
                                           'punctuation normalisation'],
                    
                    'remove_http': ['remove website address',
                                    'remove http links',
                                    'remove http',
                                    'remove www links',
                                    'clean http',
                                    'clean www',
                                    'remove links',
                                    'remove website links'
                                   ],
                    
                    'remove_numbers': ['remove numbers',
                                     'clean numbers',
                                     'remove numerical values',
                                     'clean numerical values',
                                     'get rid of numbers'],
                    
                    'remove_whitespace': ['remove whitespace',
                                         'remove white spaces',
                                         'clean white space',
                                         'clean whitespaces',
                                         'remove empty space',
                                         'remove blank spaces',
                                         'remove blanks',
                                         'clean voids',
                                         'remove voids'],
                    
                    'stemmer': ['create stemmer',
                               'make stemmer',
                               'generate stemmer',
                                'text stemming',
                               'stem text',
                                'create stem',
                                'make stem',
                                'stem',
                               'remove affixes',
                               'get rid of affixes',
                               'strip suffixes'],
                    
                    'remove_handle': ['remove twitter handle',
                                     'remove twitter',
                                     'get rid of twitter handle',
                                     'clean twitter handle',
                                     'remove twitter sign',
                                     'remove twitter mentions',
                                     'clean twitter mentions',
                                     'get rid of twitter mentions']
                    }


info_nlptxtclean = {'clean_text':{'module':'nlp_cleantext',
                                  'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'text cleaning',
                                 'input_format':'pd.Series',
                                 'description':'general cleaning of input text data, normalise text, remove punctuation, stop words and lemmatise '},

                    'lemma_text':{'module':'nlp_cleantext',
                                  'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'transform word',
                                 'input_format':'pd.Series',
                                 'description':'lemmatisation of input text data'},
                          
                    'norm_text':{'module':'nlp_cleantext',
                                  'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'transform word',
                                 'input_format':'pd.Series',
                                 'description':'lower the register of text data'},
                    
                    'convert_emoji':{'module':'nlp_cleantext',
                                  'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'text cleaning',
                                 'input_format':'pd.Series',
                                 'description':'transalte emoji icons in text data to interpretable names'},
                    

                    'remove_emoji':{'module':'nlp_cleantext',
                                  'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'text cleaning',
                                 'input_format':'pd.Series',
                                 'description':'remove emoji icons in text data'},
                    
                    
                    'remove_special_char':{'module':'nlp_cleantext',
                                  'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'text cleaning',
                                 'input_format':'pd.Series',
                                 'description':'remove special characters such as punctuation, brackets and alike from text data'},


                    'remove_http':{'module':'nlp_cleantext',
                                  'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'text cleaning',
                                 'input_format':'pd.Series',
                                 'description':'remove website links related to http and www from text'},                
                    
                    
                    'remove_numbers':{'module':'nlp_cleantext',
                                  'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'text cleaning',
                                 'input_format':'pd.Series',
                                 'description':'remove numerical values from text'},                 

                    
                    'remove_whitespace':{'module':'nlp_cleantext',
                                  'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'text cleaning',
                                 'input_format':'pd.Series',
                                 'description':'remove white space (blanks) from text data'},
                    
                    'stemmer':{'module':'nlp_cleantext',
                                'action':'text normalisation',
                                 'topic':'natural language processing',
                                  'subtopic':'text cleaning',
                                 'input_format':'pd.Series',
                                 'description':'modify the text data to its most basic/stem form'},

                    
                    'remove_handle':{'module':'nlp_cleantext',
                                    'action':'text normalisation',
                                     'topic':'natural language processing',
                                    'subtopic':'text cleaning',
                                    'input_format':'pd.Series',
                                    'description':'remove twitter account handles from text data'}     

                    
                   }

configure_nlptxtclean = {'corpus':dict_nlptxtclean,'info':info_nlptxtclean}  