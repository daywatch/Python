"""
Given a string s, return the longest palindromic substring in s.
https://leetcode.com/problems/longest-palindromic-substring/description/
"""

import typing

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1:
            raise Exception("Your string needs to have at least two chars!")

        else:
            n_gram_collection = self.n_gram(s)
            print(n_gram_collection)

            palindrome_res = []
            # get ngrams
            for ngram in n_gram_collection:
                #compare all ngrams with its list and rev-list
                ordering_list = list(ngram)
                reversed_ordering_list = list(reversed(ordering_list))
                #print(f"{ordering_list} +++ {reversed_ordering_list}")
                if ordering_list == reversed_ordering_list:
                    palindrome_res.append(ngram)
            # check the collection
            if palindrome_res != []:
                return [item for item in palindrome_res]
            else:
                return "There is no palindromic substring in this string"

    def n_gram(self,s): #starting from bigrams to the whole string
        num_range = range(2,len(s)+1)
        ngrams = []
        for num in num_range: #0:1, 1:2, 2:3
            #print(f"grams setting: {num}")
            step = num-1
            for i in range(len(s)-step):
                #print(f"i: {i}")
                ngrams.append(s[i:i+step+1])
                #print(s[i:i+step+1])
        #print(f"final: {ngrams}")
        return ngrams

solution_instance = Solution()
print(solution_instance.longestPalindrome("abba"))
print(solution_instance.longestPalindrome("bb"))