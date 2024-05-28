"""
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);
https://leetcode.com/problems/zigzag-conversion/description/
"""

class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if len(s) == 0 or numRows <= 0:
            raise Exception("Please enter a valid string and row number that are bigger than zero")
        elif len(s) == 1 and numRows == 1:
            return s
        else:
            """
            [P,0,0,I] 
            [A,0,L] 
            [Y,A,0] 
            [P,0,0] 
            
            if we only think about what list the chars belong, we can ignore the 0 fillers
            [1,NULL, NULL, 1]
            [2,NULL, 2]
            [3,3]
            [4,]
            
            so we can transform the problem as simply indexing the string like 1234321234...
            
            """
            zigzag_sequence = list(range(1,numRows+1))
            zigzag_sequence += list(reversed(zigzag_sequence))[1:-1]
            print(zigzag_sequence)

            if len(s) // len(zigzag_sequence) == 0:
                return s
            elif len(s) // len(zigzag_sequence) >0:
                whole_chunks = len(s) // len(zigzag_sequence)
                remainder = len(s) % len(zigzag_sequence)
                res_index_list = zigzag_sequence * whole_chunks + [zigzag_sequence[i] for i in range(0,remainder)]
                char_index_pairs = list(zip(s, res_index_list))

                collection = [[] for i in range(0,numRows)]
                for pair in char_index_pairs:
                    ind = pair[1]
                    collection[ind-1].append(pair[0])
                print(collection)

                final = ""
                for row in collection:
                    for char in row:
                        final += char

                return final

solution_instance = Solution()
print(solution_instance.convert("PAYPALISHIRING",3))
print(solution_instance.convert("PAY",5))
print(solution_instance.convert("P",3))