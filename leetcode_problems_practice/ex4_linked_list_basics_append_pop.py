class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        
class LinkedList:
    def __init__(self, value):
        new_node = Node(value)
        self.head = new_node
        self.tail = new_node
        self.length = 1

    def print_list(self):
        temp = self.head
        while temp is not None:
            print(temp.value)
            temp = temp.next
        print("\n")
    
    def make_empty(self):
        self.head = None
        self.tail = None
        self.length = 0
        
    def append(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1

    # pop: nothing in the node, only one node
    # you have to start from head
    def pop(self):
        if self.length == 0:
            return None
        
        else: #use TEMPORARY pre(n-1) and temp(n) so pre can track the previous ones
            pre = self.head
            temp = self.head
            while temp.next is not None:
                pre = temp
                temp = temp.next #move temp
            self.tail = pre
            self.tail.next = None
            self.length -= 1

            if self.length == 0: #after deduction of 1 for 1length; 
                # need to make sure head and tail are none
                self.head = None
                self.tail = None

            return temp.value

    #def preppend(self,value):

    #def insert(self, index, value): 
    
    # append, pop, preppend, insert, remove
my_linked_list = LinkedList(1)
my_linked_list.make_empty()

# append tests
my_linked_list.append(1)
my_linked_list.append(2)
my_linked_list.append(3)
my_linked_list.append(4)

print('Head:', my_linked_list.head.value)
print('Tail:', my_linked_list.tail.value)
print('Length:', my_linked_list.length, '\n')

print('Linked List:')
my_linked_list.print_list()

# pop tests for N, 1, and 0
print(my_linked_list.pop())
print(my_linked_list.pop())
print(my_linked_list.pop())
print(my_linked_list.pop())
print(my_linked_list.pop())


my_linked_list.print_list()