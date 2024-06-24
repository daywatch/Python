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
        if self.length == 0:
            return None
        temp = self.head
        self.head = self.head.next
        temp.next = None
        self.length -= 1
        if self.length == 0:
            self.tail = None
        return temp

    def get(self, index):
        if index < 0 or index >= self.length:
            return None
        temp = self.head
        for _ in range(index):
            temp = temp.next
        return temp
        
    def set_value(self, index, value):
        temp = self.get(index)
        if temp:
            temp.value = value
            return True
        return False
    
    def insert(self, index, value):
        if index < 0 or index > self.length:
            return False
        if index == 0:
            return self.prepend(value)
        if index == self.length:
            return self.append(value)
        new_node = Node(value)
        temp = self.get(index - 1)
        new_node.next = temp.next
        temp.next = new_node
        self.length += 1   
        return True  

    def remove(self,index):
        if self.length == 0:
            return False
        # out of range indexes
        elif index < 0 or index >= self.length:
            return None #return poped out node
        else:
            if index == 0: #pop_first
                temp = self.head
                self.head = temp.next
                temp.next = None
            elif index == self.length-1:#pop_last
                temp = self.head
                pre = self.head
                while temp.next != None:
                    pre = temp
                    temp = temp.next
                pre.next = None
                self.tail = pre
            else: #pop in the mid
                temp = self.head
                pre = self.head
                post = self.head.next
                while post.next != None:
                    post = post.next
                    pre = temp
                    temp = temp.next
                pre.next = post
                temp.next = None
                self.length -= 1
                return temp.value
                


my_linked_list = LinkedList(1)
my_linked_list.append(3)
my_linked_list.append(32)
my_linked_list.append(99)

my_linked_list.remove(2)
my_linked_list.print_list()
print('final Length:', my_linked_list.length, '\n')