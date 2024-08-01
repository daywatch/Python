# breath first search - row by row in a tree
# queue: store the entire node
# results: only traversed value
# condition: while queue is not empty

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
    

    def BFS(self):
        current_node = self.root
        queue = []
        results = []
        
        queue.append(current_node)

        while len(queue) > 0:
            current_node = queue.pop(0) #always pop the first node off

            results.append(current_node.value)

            if current_node.left is not None: # look at both sides
                queue.append(current_node.left)
            if current_node.right is not None:
                queue.append(current_node.right)

        return results

my_tree = BinarySearchTree()
my_tree.insert(47)
my_tree.insert(21)
my_tree.insert(76)
my_tree.insert(18)
my_tree.insert(27)
my_tree.insert(52)
my_tree.insert(82)

print(my_tree.BFS())
            




