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

    def pop(self):
        if self.length == 0:
            return None
        temp = self.head
        pre = self.head
        while(temp.next):
            pre = temp
            temp = temp.next
        self.tail = pre
        self.tail.next = None
        self.length -= 1
        if self.length == 0:
            self.head = None
            self.tail = None
        return temp

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
    
    def pop_first(self):
        #leng=0
        if self.length == 0:
            return None
        #leng=1
        elif self.length == 1:
            self.head = None
            self.tail = None
        #leng>1
        else:
            first_old = self.head
            second_old = first_old.next
            self.head = second_old #CUTOFF
            first_old.next = None
            self.length -= 1
            return first_old.value
        
    def get(self,index):
        if self.length == 0:
            return None
        if index <0 or index >= self.length:
            return "invalid index"
        else:
            temp = self.head
            for _ in range(index): #use given index, not the whole length of the list
                temp = temp.next
            return temp.value


my_linked_list = LinkedList(0)
my_linked_list.append(1)
my_linked_list.append(2)
my_linked_list.append(3)
#my_linked_list.print_list()
print("f")
print(my_linked_list.get(0))