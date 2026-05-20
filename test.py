""" This file contains some test code for the functions in the main code. It is not meant to be run as a script, but rather to be used as a reference for how to use the functions and to test their functionality.
It is also a good place to test any new functions that are added to the main code before integrating them into the main codebase.
"""
import re
import sys

from collections import Counter
from datetime import datetime, timezone

import pandas

from tree_node import TreeNode

def create_staircase(nums):
  step = 1
  subsets = []
  while len(nums) != 0:
    if len(nums) >= step:
      subsets.append(nums[0:step])
      nums = nums[step:]
      step += 1
    else:
      return False
      
  return subsets

def insert_text_into_text(target: str, text_to_insert: str, position: int) -> str:
  if position < 0 or position > len(target):
    raise ValueError("Position must be between 0 and the length of the target string.")
  return target[:position] + text_to_insert + target[position:]

def normalize_data(df: pandas.DataFrame) -> pandas.DataFrame:
  df_new = df.copy()
  df_new["dob"] = pandas.to_datetime(df_new["dob"], format="%Y-%m-%d", errors="coerce")
  df_new["name"] = df_new["name"].apply(normalize_name)
  df_new["postal"] = df_new["postal"].apply(lambda p: insert_text_into_text(re.sub(r"\s+", "", p.strip()).upper() if p else "", " ", 3))
  return df_new

def normalize_name(name: str) -> str:
  name = re.sub(r"\s+", " ", name.strip()).upper()
  return name

CLINICS = {"CLINIC", "HOSPITAL", "MEDICAL", "HEALTH", "CARE", "CENTER", "CENTRE", "OFFICE", "CLINIQUE", "CLINICA", "CLINIQUE", "CLINIC"}
def looks_like_person_name(name: str) -> bool:
  tokens = re.sub(r"[^A-Z0-9\s]", " ", name.upper()).split() if name else []
  if len(tokens) < 2:
    return False
  if any(token in CLINICS for token in tokens):
    return False

  return True

if __name__ == "__main__":
  print(create_staircase([1, 2, 3, 4, 5, 6]))
  print(create_staircase([1, 2, 3, 4, 5]))
  print(create_staircase([1, 2, 3, 4, 5, 6, 7]))

  print(f"#Test how DataFrame works with the normalize_data function")
  df = pandas.DataFrame([{"name": "Jason", "dob": "1998-2-3", "postal": "l6c"}, {"name": " Karen  T ", "dob": "2009-4-1", "postal": "L3c4M5"}, {"name": "Dr. Priya Sharma, MD", "dob": "1980-12-15", "postal": "M5G 1X 8 "}])
  df_new = normalize_data(df)
  print(df_new)

  print(f"#Test looks_like_person_name function")
  print(looks_like_person_name("Jas -  on= "))
  print(looks_like_person_name("Jason-clinic "))
  print(looks_like_person_name("Health center"))
  print(looks_like_person_name("Dr. Karen T Denal Office"))
  print(looks_like_person_name("Karen Care"))

  print(f"#Test Regular Expression")
  def clinic_len_or_invalid(clinic: str) -> tuple[int, str]:
    if re.match(r"[^a-zA-Z0-9\s]+", clinic):
      return (0, clinic)
    else:
      return (-len(clinic), clinic)

  _CLINIC_RE = re.compile(r"\b(?:" + "|".join(re.escape(clinic) for clinic in sorted(CLINICS, key=clinic_len_or_invalid, reverse=False)) + r")\.?\b", re.IGNORECASE)
  s = " Jason-clinic Center"
  print(_CLINIC_RE.findall(s))

  def topKFrequent(nums, k):
    print(datetime.now(timezone.utc).isoformat());
    countDict = {}
    for item in nums:
      countDict[item] = countDict.get(item,0) + 1    
    #sortedCounts = sorted(countDict, key=lambda x: countDict[x], reverse=True)
    buckets = [[] for _ in range(len(nums) + 1)]
    for n, c in countDict.items():
        buckets[c].append(n)
        
    # 3. Iterate through buckets from highest frequency to lowest
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        for n in buckets[i]:
            result.append(n)
            if len(result) == k:
                break

    print(datetime.now(timezone.utc).isoformat());

    return result

  print(topKFrequent([1,1,456,3,2,57,4,5,2,7,8,9,9,9,9,1023,46,7,88899999,532,12,12,12,243,3512,23,2123,123,123,3,5,5,676,73,769,3,2,4,56,78,2,2,2,3,8,23,45,687,126,845,1,2,2,3], 5))

  root = TreeNode(3)
  root.left = TreeNode(5)
  root.right = TreeNode(1)
  root.left.left = TreeNode(6)
  root.left.right = TreeNode(2)
  root.right.left = TreeNode(0)
  root.right.right = TreeNode(8)
  root.left.right.left = TreeNode(7)
  root.left.left.left = TreeNode(11)
  root.left.right.right = TreeNode(4)
  root.left.right.right.left = TreeNode(9)
  a = root.left.left.left
  b = root.left.right.right.left
  TreeNode.print_tree(root)
  print(TreeNode.lowest_common_ancestor(root, a, b).value)
  a = root.left.left.left
  b = root.right.right
  print(TreeNode.lowest_common_ancestor(root, a, b).value)
