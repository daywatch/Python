input_list = [4,2,6,5,1,3]

"""
for each ith element in list:
    min_index = i

    for each jth in the ith element
        compare each array[min_index] and array[i+j+1] and renew min_index

    swap between ith and new min_index

"""
def selection(array):

    for i in range(len(array)-1):

        min_index = i
        current = array[i]

        for j in range(len(array)-i-1): #OR: in range(i+1,len(array)) and then use j instead of j+1
            
            if array[min_index] > array[i+j+1]:

                min_index = i+j+1
        
        array[i] = array[min_index]
        array[min_index] = current

    return array

print(selection(input_list))