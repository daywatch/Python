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
    
    def reverse(self):
        #nothing to reverse
        if self.length == 0:
            return None
        #only one node
        elif self.length == 1:
            return self
        else:
            current_pointer = self.head
            insertion_pointer = self.tail
            
            # round1: create an insertion gap
            current_copy = current_pointer
            current_pointer = current_pointer.next
            current_copy.next = None
            insertion_pointer.next = current_copy
            insertion_pointer_plus1 = insertion_pointer.next
            self.tail = current_copy
            self.head = insertion_pointer

            #round 2-N: 
            for _ in range(2,self.length):
                current_copy = current_pointer
                current_pointer = current_pointer.next
                current_copy.next = insertion_pointer_plus1
                insertion_pointer.next = current_copy
                insertion_pointer_plus1 = current_copy
            return True
        
    def reverse_solution2(self):
        # the main point is to reverse all arrows!
        # reverse H and T
        temp = self.head
        self.head = self.tail
        self.tail = temp

        after = temp.next
        before = None

        for _ in range(self.length):
            after = temp.next
            temp.next = before #flip arrow
            before = temp
            temp = after # need to avoid the gap
        return True

my_linked_list = LinkedList(1)
my_linked_list.append(344)
my_linked_list.append(5)
my_linked_list.append(995)

my_linked_list.reverse()
my_linked_list.reverse_solution2()
my_linked_list.print_list()
#print('final Length:', my_linked_list.length, '\n')


