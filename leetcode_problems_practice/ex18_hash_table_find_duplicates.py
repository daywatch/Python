
# WRITE FIND_DUPLICATES FUNCTION HERE #
def find_duplicates(input_list):
    if type(input_list) != list or len(input_list) < 2:
        return []
    else:
        my_dict = {}
        dup = []
        
        for item in input_list:
            if item in my_dict:
                if item not in dup:
                    dup.append(item)
            else:
                my_dict[item] = True
        return dup
    
#######################################




print ( find_duplicates([1, 2, 3, 4, 5]) )
print ( find_duplicates([1, 1, 2, 2, 3]) )
print ( find_duplicates([1, 1, 1, 1, 1]) )
print ( find_duplicates([1, 2, 3, 3, 3, 4, 4, 5]) )
print ( find_duplicates([1, 1, 2, 2, 2, 3, 3, 3, 3]) )
print ( find_duplicates([1, 1, 1, 2, 2, 2, 3, 3, 3, 3]) )
print ( find_duplicates([]) )



"""
    EXPECTED OUTPUT:
    ----------------
    []
    [1, 2]
    [1]
    [3, 4]
    [1, 2, 3]
    [1, 2, 3]
    []

"""

