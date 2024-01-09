import re

'''

Check if string contains a float value

'''

def isfloat(strs:str):
  if(re.match(r'^-?\d+(?:\.\d+)$', strs) is None):
    return False
  else:
    return True