"""
Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).

"""

"""
rules:
1.ungrammtical: nothing but *
2.no reg
3.* for zero string
4.partial coverage?
"""

import re
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if p == "*":
            raise Exception("You need to put preceding chars before any *")
        if p == "" and s != "":
            raise Exception("There is no pattern to compare against the example")
        if "*" in p or "." in p:
            res = re.search(p,s)
            if res.group(0) == s:
                return "true"
            else: return "false"
        else:
            if p == s: return "true"
            else: return "false"


solution_instance = Solution()
print(solution_instance.isMatch("aa","a."))
print(solution_instance.isMatch("ab",".*"))
print(solution_instance.isMatch("aa","a"))