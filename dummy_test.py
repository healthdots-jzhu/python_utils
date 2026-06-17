import tempfile
from pathlib import Path
from pdf_writer import write_pdf

def test_pdf_tools():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_input_file = Path(temp_dir) / "test_input1.txt"
        test_input_file.write_text(
            f"""
<html>
    <head><meta charset='UTF-8' /></head>
    <body>
        <h1>Hellow world</h1>
        <div>
            <table>
                <tr><th>Name</th><th>Score</th></tr>
                <tr><td>Alice</td><td>90</td></tr>
                <tr><td>Bob</td><td>85</td></tr>
            </table>
        </div>
    </body>
</html>""",
            encoding="utf-8"
        )
        
        write_pdf(Path(temp_dir) / "test_input1.txt", output_path=Path(r"c:\temp\test_output1.pdf"))
        write_pdf(Path(r"c:\temp\da_project.html"), output_path=Path(r"c:\temp\da_project.pdf"))

def longest_substring_without_repeating_characters(s: str) -> tuple[int, str]:
    char_map = {}
    max_length = 0
    left = 0
    longest_substring = ""
    for right, char in enumerate(s):
        if (char in char_map) and char_map[char] >= left:
            left = char_map[char] + 1
        char_map[char] = right
        if (right - left + 1 > max_length):
            max_length = right - left + 1
            longest_substring = s[left:right+1]
    return (max_length, longest_substring)

if __name__ == "__main__":
    #test_pdf_tools()
    length, substring = longest_substring_without_repeating_characters("abcabcbb")
    print(f"Longest substring without repeating characters: '{substring}' with length {length}")

