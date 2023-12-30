
# helper function to get bracket pairs
def get_bracket_content(lst:list):

    # lst format
    # ['A','B','{','B','}','D']

    stack = []; pairs = []
    for i, item in enumerate(lst):
        if item == '{':
            stack.append(i)
        elif item == '}':
            if stack:
                pairs.append((stack.pop(), i))
            else:
                print("[note]: Unmatched closing bracket at index", i)
    if stack:
        print("[note]: Unmatched opening bracket(s) at index", stack)
    
    return pairs