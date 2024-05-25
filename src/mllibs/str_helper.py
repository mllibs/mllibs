import re

'''

Check if string contains a float value

'''

def isfloat(strs:str):
	if(re.match(r'^-?\d+(?:\.\d+)$', strs) is None):
		return False
	else:
		return True

'''

Check the input string for storage pattern

'''

def extract_word_in_brackets(input_string:str):

	word_in_brackets = None

	# Define a regular expression pattern to match 'as' followed by words inside brackets
	pattern = r'as \[([^\]]+)\]|\[([^\]]+)\]'

	# Find all occurrences of the pattern in the input string
	matches = re.findall(pattern, input_string)

	# Extract the word inside brackets and remove 'as' and the brackets from the input string
	for match in matches:
			word_in_brackets = match[0] if match[0] else match[1]
			input_string = re.sub(r'as \[' + re.escape(word_in_brackets) + r'\]', '', input_string)
			input_string = re.sub(r'\[' + re.escape(word_in_brackets) + r'\]', '', input_string)

	return word_in_brackets, input_string