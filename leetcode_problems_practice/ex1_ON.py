"""
<> Big O: O(n)
Implement a Python function called print_items.

This function should take a single integer as an argument and print out a sequence of numbers from 0 up to, but not including, the provided integer.

The function should use a for loop and Python's built-in range function to generate the sequence of numbers.

The function signature should be: def print_items(n):


Example:

Here is an example of how your function should behave:


If you call print_items(10), your function should print:

0
1
2
3
4
5
6
7
8
9
"""

def print_items(input_integer):
    if type(input_integer) != int or input_integer == 0:
        return "You need to give an int bigger than 0"
    else:
        for i in range(input_integer):
            print(i)
      
        
# DO NOT CHANGE THIS LINE:
print_items(10)