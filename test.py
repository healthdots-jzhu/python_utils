""" This file contains some test code for the functions in the main code. It is not meant to be run as a script, but rather to be used as a reference for how to use the functions and to test their functionality.
It is also a good place to test any new functions that are added to the main code before integrating them into the main codebase.
"""
import re
import pandas

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
