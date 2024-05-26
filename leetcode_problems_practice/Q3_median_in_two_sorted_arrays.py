"""
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
https://leetcode.com/problems/median-of-two-sorted-arrays/description/
"""

from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        merged_list = sorted(nums1 + nums2)

        length = len(merged_list)

        if length % 2 == 1:
            return float(merged_list[int((length+1)/2)-1])

        if length % 2 == 0:
            ind1 = int(length/2)-1
            print(ind1)
            ind2 = int(length / 2)
            print(ind2)

            return ( merged_list[ind1] + merged_list[ind2] ) /2

solution_instance = Solution()
print(solution_instance.findMedianSortedArrays([1,3],[2]))
print(solution_instance.findMedianSortedArrays([1,3,4],[2,10,11]))
print(solution_instance.findMedianSortedArrays([1,2],[3,4]))