"""
Given a string s, find the length of the longest
substring without repeating characters
https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
"""
from typing import List

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        #chunk the string into substrings
        sub_collection = []
        s_list = list(s)
        sub = ""
        s_copy = list(s)
        while s_copy != []: #every iteration makes the string pop out 1st element
            if s_copy[0] not in sub:
                sub+= s_copy[0]
            else:
                sub_collection.append(sub)
                sub = s_copy[0]

            s_copy.pop(0)

            #exception: the last element needs to be appended
            if len(s_copy) == 0:
                sub_collection.append(sub)

        length_list = [len(sub) for sub in sub_collection]

        return max(length_list)


solution_instance = Solution()
print(solution_instance.lengthOfLongestSubstring("1231234"))
print(solution_instance.lengthOfLongestSubstring("bbbb"))
print(solution_instance.lengthOfLongestSubstring("pwwkew"))