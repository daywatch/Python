input_list = [4,2,6,5,1,3] 

"""
for each element, look backward and renew the order
note: 1.backward looking might have an out of range issue
"""
def insertion1(array):
    for i in range(len(array)):
        for j in range(i, 0, -1):
            if array[j] < array[j - 1]:
                # Swap elements
                array[j], array[j - 1] = array[j - 1], array[j]
    return array


def insertion2(array):
    for i in range(1,len(array)):
        temp = array[i]
        j = i-1
        while temp < array[i-1] and j>-1: #to avoid out of range
            array[i] = array[i-1]
            array[i-1] = temp
            j -= 1
    return array

print(insertion1(input_list))
print(insertion2(input_list))