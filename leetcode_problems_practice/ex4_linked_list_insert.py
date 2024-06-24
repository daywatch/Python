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
    
    def make_empty(self):
        self.head = None
        self.tail = None
        self.length = 0
        
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
    
    def prepend(self, value):
        new_node = Node(value)
        if self.length == 0:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
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
       
    def insert(self,index,value):
        if self.length == 0:
            print("there is no node, so this insertion means a new node with this value")
            return Node(value).value
        else:
            if index < 0  or index > self.length:
                return "invalid index"
            else:
                if index == 0:
                    return self.prepend(value)
                if index == self.length:
                    return self.append(value)
                else:
                    new_node = Node(value)
                    temp = self.head
                    for _ in range(index-1):
                        temp = temp.next

                    #the ordering is important
                    new_node.next = temp.next
                    temp.next = new_node 
                    self.length += 1

                    return True           


my_linked_list = LinkedList(0)
#my_linked_list.make_empty()

my_linked_list.append(1)
my_linked_list.append(2)
my_linked_list.append(3)

print("f")
print(my_linked_list.insert(1,10))
my_linked_list.print_list()