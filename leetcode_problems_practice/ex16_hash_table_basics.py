# hash: always have prime numbers of addresses
# use list to model the hash table
# O(1) because lots of sparse indices

class HashTable:
    def __init__(self,size=7):
        self.data_map = [None] * size

    def _hash(self,key):#address computer
        my_hash = 0
        for letter in key:
            #23 is a prime number
            my_hash = (my_hash + ord(letter)*23)%len(self.data_map)
        return my_hash
    
    def print_table(self):
        for i,val in enumerate(self.data_map):
            print(i, ": ", val)
    
    def set_item(self,key,value):
        index = self._hash(key)
        if self.data_map[index] == None: #initialize []
            self.data_map[index] = []
        self.data_map[index].append([key,value])
    
    def get_item(self,key):
        index = self._hash(key)
        record = self.data_map[index]
        if record is not None:
            for item in record:
                if item[0] == key:
                    return item[1]
        return None
    
    def keys(self):
        all_keys = []
        for record in self.data_map:
            if record != None:
                all_keys.append(record[0][0])
        return all_keys
                


hash = HashTable()
hash.print_table()

hash.set_item("bolt",1400)
hash.set_item("books",1)
hash.set_item("brushes",-23)
print("\n")
hash.print_table()
print("\n")
print(hash.get_item("books"))
print(hash.get_item("sky"))
print("\n")
print(hash.keys())

