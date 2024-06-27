# for any given two sorted lists, use merge() to combine them into one
def merge(list1,list2): #log(n)
    merged = []
    i = 0
    j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged.append(list1[i])
            i += 1
        else: 
            merged.append(list2[j])
            j += 1

    if i == len(list1): #when list1 is empty
        merged += list2[j:]
    if j == len(list2):
        merged += list1[i:]

    return merged

array1 = [1,3,7,8]
array2 = [2,4,5,6]
#print(merge(array1,array2))

#on top of merge(), we need to convert a list into single-value lists
def merge_sort(array):
    fully_splitted_lists = [[item] for item in array]

    if len(fully_splitted_lists)%2 == 0: #even numbers
        while len(fully_splitted_lists) != 1:
            
            # subgroup values as pairs within the list
            new = [(fully_splitted_lists[i], fully_splitted_lists[i + 1]) for i in range(0, len(fully_splitted_lists), 2)]

            #perform merge() and append the output back the list and renew the list
            new2 = []
            for item in new:
                new2.append(merge(item[0],item[1]))

            fully_splitted_lists = new2

    return fully_splitted_lists[0]

input_list = [4,5,7,1,3,2,8,6]
print(merge_sort(input_list))


# another version: mid index and recursion
def merge_sort(my_list): #n*logn
    if len(my_list) == 1:
        return my_list
    mid_index = int(len(my_list)/2)
    left = merge_sort(my_list[:mid_index]) #keep splitting
    right = merge_sort(my_list[mid_index:])
    
    return merge(left, right)



original_list = [3,1,4,2]

sorted_list = merge_sort(original_list)

print('Original List:', original_list)

print('\nSorted List:', sorted_list)
