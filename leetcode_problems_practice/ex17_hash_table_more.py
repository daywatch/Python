"""
1,3,5
2,4,5
look through what in common

Q: find a more efficient way instead of O(n^2)
A: use a dictionary to search O(2n)
"""
def item_in_common(list1,list2):
    my_dict = {}
    for i in list1:
        my_dict[i] = True
    
    for j in list2:
        if j in my_dict:
            return True
    return False

print(item_in_common([1,3,5],[2,4,5]))