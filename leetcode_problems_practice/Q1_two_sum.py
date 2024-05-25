"""
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

https://leetcode.com/problems/two-sum/description/

"""
from typing import List

class Solution:

    def __init__(self,nums: List[int], target: int):
        self.nums = nums
        self.target = target
        print(f"the original list is {self.nums}")
    def twoSum(self) -> List[int]:
        res_scope = []
        for i in range(len(self.nums)):
            for j in range(len(self.nums)):
                res_scope.append([[self.nums[i],self.nums[j]],self.nums[i]+self.nums[j]])

        # delete the reversed cases
        reduced_scope = []
        for item in res_scope:
            if item not in reduced_scope:
                reduced_scope.append(item)
        print(reduced_scope)

        #tell whether the target matches
        matches = []
        for i in range(len(reduced_scope)):
            if reduced_scope[i][1] == self.target:
                matches.append(reduced_scope[i])

        #collect multiple matches
        match_results = []
        for match in matches:
           input_matches = match[0]
           ind1 = [i for i, x in enumerate(self.nums) if x == input_matches[0]]
           ind2 = [i for i, x in enumerate(self.nums) if x == input_matches[1]]
           print(ind1)

           #consider multiple index matches
           if len(ind1) == 1:
               match_results.append([ind1[0],ind2[0]])
           else:
               match_results.append(ind1)

        # clean up the index results
        match_results = [set(item) for item in match_results]
        final = []
        for item in match_results:
            item = list(item)
            if len(item) == 2 and item not in final:
                final.append(item)
        return final


solution = Solution(nums=[3,3],target=6)
print(solution.twoSum())

solution = Solution(nums=[1,2,3],target=5)
print(solution.twoSum())

solution = Solution(nums=[3,2,4],target=6)
print(solution.twoSum())