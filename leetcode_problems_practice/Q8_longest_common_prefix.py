"""
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".
"""
from typing import List

class solution():
    def longest_common_prefix(self,str_array: List[str]) -> str :
        # same strings
        if list(set(str_array)) ==1:
            return str_array[0]

        else:
            str_array_leng = [len(string) for string in str_array]
            min_ind = str_array_leng.index(min(str_array_leng))
            str_min = str_array[min_ind]
            #print(str_min)

            prefix_candidates = [str_min[0:i] for i in range(0,len(str_min))][1:]
            res = []

            while prefix_candidates != []:

                cand = prefix_candidates[0]
                mapped_string_cand = [string[0:len(cand)] for string in str_array]
                prefixes_current = list(set(mapped_string_cand))
                if len(prefixes_current) == 1 and prefixes_current[0] == cand:
                    res.append(mapped_string_cand[0])

                prefix_candidates = prefix_candidates[1:]

            if res == []:
                return "no shared prefix"
            else:
                return f"The longest shared prefix is {res[-1]}"

            return prefix_candidates



        return str_array

solution_instance = solution()
print(solution_instance.longest_common_prefix(["12345","1234"]))
print(solution_instance.longest_common_prefix(["dog","racecar","car"]))
print(solution_instance.longest_common_prefix(["flower","flow","flight"]))

