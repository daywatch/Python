class Node:
    def __init__(self,value):
        self.value = value
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self,value):
        new_node = Node(value)
        self.head = new_node
        self.tail = new_node
        self.length = 1

    def print_list(self):
        temp = self.head
        while temp is not None:
            print(temp.value)
            temp = temp.next

    def append(self,value):
        new_node = Node(value)
        if self.length == 0:
            self.head = new_node
            self.tail = new_node
        else:
            tail_old = self.tail
            tail_old.next = new_node
            new_node.prev = tail_old
            self.tail = new_node
        self.length += 1
        return True
    
    def pop(self):
        # nothing to pop
        if self.length == 0:
            return None
        elif self.length == 1:
            return None
        else:
            pop = self.tail
            self.tail = pop.prev
            self.tail.next = None
            pop.prev = None
            self.length -= 1
            return pop.value
    
    def prepend(self,value):
        new_node = Node(value)
        if self.length == 0:
            return new_node
        else:
            temp = self.head
            new_node.next = temp
            temp.prev = new_node
            self.head = new_node
        self.length += 1
        return True
    
    def pop_first(self):
        if self.length == 0:
            return None
        elif self.length == 1:
            self.head = None
            self.tail = None
            self.length -= 1
        else:
            temp = self.head
            self.head = self.head.next
            temp.next = None
            self.head.prev = None
            self.length -= 1
        return True
    
    def get(self,index):
        if self.length == 0:
            return None
        if index < 0 or index >= self.length:
            return None
        else:
            temp = self.head
            for _ in range(index):
                temp = temp.next
            return temp.value
    
    # a more efficient version of get!!!
    def get(self,index):
        if self.length == 0:
            return None
        if index < 0 or index >= self.length:
            return None
        else:
            temp = self.head
            # logN
            if index < self.length/2:
                for _ in range(index):
                    temp = temp.next
            else:
                temp = self.tail
                for _ in range(self.length-1,index,-1):
                    tempe = temp.prev
            return temp
    
    def set_value(self,index,value):
        if index < 0 or index > self.length:
            return None
        temp = self.get(index)
        if temp:
            temp.value = value
            return True
        else:
            return False
    
    def insert(self,index,value):
        new_node = Node(value)
        if self.length == 0:
            self.head = new_node
            self.tail = new_node
            self.length += 1
        
        if index == 0:
            temp = self.head
            temp.prev = new_node
            new_node.next = temp
            self.head = temp.prev
            self.length += 1

        elif index == self.length -1:
            temp = self.tail
            temp.next = new_node
            new_node.prev = temp
            self.tail = temp.next
            self.length += 1
        
        else:
            before = self.get(index-1)
            after = before.next
            before.next = new_node
            new_node.prev = before
            new_node.next = after
            after.prev = before
            self.length += 1
        return True

    def remove(self, index):
        if index < 0 or index >= self.length:
            return None
        if index == 0:
            return self.pop_first()
        if index == self.length - 1:
            return self.pop()

        temp = self.get(index)
        
        temp.next.prev = temp.prev
        temp.prev.next = temp.next
        temp.next = None
        temp.prev = None

        self.length -= 1
        return temp


my_doubly_linked_list = DoublyLinkedList(7)
my_doubly_linked_list.print_list()

print("\nappend")
my_doubly_linked_list.append(15)
my_doubly_linked_list.append(100)
my_doubly_linked_list.print_list()

print("\npop")
my_doubly_linked_list.pop()
my_doubly_linked_list.print_list()

print("\nprepend")
my_doubly_linked_list.prepend(90)
my_doubly_linked_list.print_list()

print("\npop_first")
my_doubly_linked_list.pop_first()
my_doubly_linked_list.print_list()

print("\nget")
print(my_doubly_linked_list.get(1).value)

print("\nset")
print(my_doubly_linked_list.set_value(1,1444))
my_doubly_linked_list.print_list()

print("\ninsert")
my_doubly_linked_list.append(100)
print(my_doubly_linked_list.insert(1,5678))
my_doubly_linked_list.print_list()

print("\nremove")
print(my_doubly_linked_list.remove(3))
my_doubly_linked_list.print_list()


