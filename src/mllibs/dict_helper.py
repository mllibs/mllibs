
def sfp(args:dict,preset:dict,key:str):
    
    # try string eval else its not string
    # alternatively choose from default dict
    if(args[key] is not None):
        try:
            return eval(args[key])
        except:
            return args[key]
    else:
        return preset[key]  

def sfpne(args:dict,preset:dict,key:str):
    
    if(args[key] is not None):
        return args[key]
    else:
        return preset[key]  
    
def sgp(args:dict,key:str):
    
    if(args[key] is not None):
        return eval(args[key])
    else:
        return None

'''

Check two subset column names [column] & [col]

'''

def column_to_subset(args:dict):

    if(args['column'] == None and args['col'] == None and args['columns'] == None):
        return None
    else:

        if(args['column'] != None):
            if(isinstance(args['column'],str) == True):
                return [args['column']]      
            elif(isinstance(args['column'],list) == True):
                return args['column']
            else:
                print('[note] error @column_to_subset')

        elif(args['columns'] != None):
            if(isinstance(args['columns'],str) == True):
                return [args['columns']]      
            elif(isinstance(args['columns'],list) == True):
                return args['columns']
            else:
                print('[note] error @column_to_subset')

        elif(args['col'] != None):  
            if(isinstance(args['col'],str) == True):
                return [args['col']]      
            elif(isinstance(args['col'],list) == True):
                return args['col']
            else:
                print('[note] error @column_to_subset')



'''

# for converting numeric text into int/float

'''

def convert_str_to_val(args:dict,key:str):
    try:
        try:
            val = eval(args[key]) # if args[key] is a string
        except:
            val = args[key]  # else just a value
    except:
        val = None
    return val


'''

    Sort a dictionary by its values

'''

def sort_dict_by_value(data:dict):
    sorted_dict = dict(sorted(data.items(), key=lambda item: item[1]))
    return sorted_dict


'''

    Print Dictionary

'''

def print_dict(dct:dict):
    for item, amount in dct.items():
        print("{} ({})".format(item, amount))