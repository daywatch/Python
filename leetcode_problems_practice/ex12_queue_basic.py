# FIFO: first in, first out
# add - easy; remove - O(n)
# first > head; last > tail
# its the best to add from tail and remove from the head

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class Queue:
    def __init__(self,value):
        new_node = Node(value)
        self.first = new_node
        self.last = new_node
        self.length = 1

    def print_queue(self):
        temp = self.first
        while temp:
            print(temp.value)
            temp = temp.next

    def enqueue(self,value):
        new_node = Node(value)
        if self.length == 0:
            self.first = new_node
            self.last = new_node
        else:
            temp = self.last
            temp.next = new_node
            self.last = new_node
        self.length+=1
        return True
    
    def dequeue(self):
        if self.length == 0:
            return None
        if self.length == 1:
            self.first = None
            self.last = None
        else:
            temp = self.first
            self.first = temp.next
            temp.next = None
            return temp
        self.length -= 1
        return True
            

my_q = Queue(4)
my_q.print_queue()
print("\n")
my_q.enqueue(5)
my_q.enqueue(6)
my_q.print_queue()
print("\n")
my_q.dequeue()
my_q.dequeue()
my_q.print_queue()