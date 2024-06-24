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
        
    def append(self, value):
        new_node = Node(value)
        if self.length == 0:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1
        return True
    
    def set_value(self,index,value):
       if index < 0 or index > self.length:
           return "invalid index"
       else:
           # or just: temp = self.get(index) & think about None
           temp = self.head
           for _ in range(index):
               temp = temp.next
           temp.value = value
           return True
           


my_linked_list = LinkedList(0)
my_linked_list.append(1)
my_linked_list.append(2)
my_linked_list.append(3)
#my_linked_list.print_list()
print("f")
print(my_linked_list.set_value(0,10))
my_linked_list.print_list()