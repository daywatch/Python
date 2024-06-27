# bubble sort
array = [2,4,5,1,3,6]

# bubble sort: do pairwise comparisions for each num

def bubble(input_list):
    for i in range(len(input_list)-1,0,-1): #5: counting down from 5
        print(i)
        for j in range(i): #
            if input_list[j] > input_list[j+1]:
                temp = input_list[j]
                input_list[j] = input_list[j+1]
                input_list[j+1] = temp
    return input_list

print(bubble(array))
