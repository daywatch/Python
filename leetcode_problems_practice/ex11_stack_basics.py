# analogy of stack: a cup of tennis balls
# LIFO: last in first out -> O(1)

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class Stack:
    def __init__(self,value):
        new_node = Node(value)
        self.top = new_node
        self.height = 1 #stack is vertical

    def print_stack(self):
        temp = self.top
        while temp:
            print(temp.value)
            temp = temp.next
        print("\n")
    
    def push(self,value):
        new_node = Node(value)
        if self.height == 0:
            self.top = new_node
            self.height = 1 #stack is vertical
        else:
            temp = self.top
            new_node.next = temp
            self.top = new_node
            self.height += 1
        return True
    
    def pop(self):
        if self.height == 0:
            return None
        if self.height >= 1:
            temp = self.top
            self.top = temp.next
            temp.next = None
            return temp


my_stack = Stack(4)
my_stack.print_stack()

print("push")
my_stack .push(5)
my_stack .push(6)
my_stack .push(7)
my_stack.print_stack()

print("pop")
my_stack.pop()
my_stack.print_stack()
