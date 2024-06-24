"""
Stack: Implement Stack Using a List ( ** Interview Question)
In the Stack: Intro video we discussed how stacks are commonly implemented using a list instead of a linked list.

Create a constructor for Class Stack that implements a new stack with an empty list called stack_list.

Q1: use [] as stack_list
Q2: push
Q3: pop

"""

class Stack:
    def __init__(self):
        self.stack_list = []
        
    def print_stack(self):
        for i in range(len(self.stack_list)-1, -1, -1):
            print(self.stack_list[i])


    # WRITE PUSH METHOD HERE #
    def push(self,value):
        if len(self.stack_list) == 0:
            self.stack_list = [value]
        else:
            self.stack_list = self.stack_list + [value]
        return True
    ##########################



class Stack:
    def __init__(self):
        self.stack_list = []

    def print_stack(self):
        for i in range(len(self.stack_list)-1, -1, -1):
            print(self.stack_list[i])

    def is_empty(self):
        return len(self.stack_list) == 0

    def peek(self):
        if self.is_empty():
            return None
        else:
            return self.stack_list[-1]

    def size(self):
        return len(self.stack_list)

    def push(self, value):
        self.stack_list.append(value)

    # WRITE POP METHOD HERE #
    def pop(self):
        if self.is_empty():
            return None
        if self.size() == 1:
            pop = self.stack_list[0]
            self.stack_list = []
        else:
            pop = self.stack_list[-1]
            self.stack_list = self.stack_list[:-1]
        return pop
    #########################
