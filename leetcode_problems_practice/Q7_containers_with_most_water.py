"""
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.
"""

from typing import List

class Solution:
    def maxArea(self, height: List[int]) -> int:

        # zero or negative height
        if len(height) == 0 or len(list(filter(lambda x: x < 0, height))) >0:
            return ("The input list has no positive height values or has 0 in it")
        else:
            #same height values only
            if len(list(set(height))) == 1:
                return "you need to put more than one kinds of heights"
            else:
                water_dim = []
                for i in range(0,len(height)):
                    for j in range(0,len(height)):
                        if i != j:
                            print([(i,height[i]),(j,height[j])])
                            water_height = min([height[i],height[j]])
                            print(water_height)
                            water_weigth = abs(i-j)
                            print(water_weigth)
                            water_dim.append(water_weigth * water_height)
                water_dim = max(list(set(water_dim)))
                return water_dim

solution_ins = Solution()
print(solution_ins.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))
print(solution_ins.maxArea([1, 1]))