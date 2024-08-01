"""
binary search tree: for a node, everything bigger/smaller is on the right/left
if the child-node is filled get to the lower child and do the same comparison
"""

class Node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        new_node = Node(value)
        if self.root is None:
            self.root = new_node
            return True
        temp = self.root
        while (True):
            if new_node.value == temp.value:
                return False
            if new_node.value < temp.value:
                if temp.left is None:
                    temp.left = new_node
                    return True
                temp = temp.left
            else: 
                if temp.right is None:
                    temp.right = new_node
                    return True
                temp = temp.right



my_tree = BinarySearchTree()

my_tree.insert(3)
my_tree.insert(1)
my_tree.insert(5)

print(my_tree.root.value)
print(my_tree.root.left.value)
print(my_tree.root.right.value)
