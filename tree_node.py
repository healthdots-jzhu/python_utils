class TreeNode:
  def __init__(self, value, left = None, right = None):
    self.value = value
    self.left = left
    self.right = right

  @staticmethod
  def build_display(node):
      text = str(node.value)
      text_width = len(text)

      if node.left is None and node.right is None:
        return [text], text_width, text_width // 2

      if node.right is None:
        left_lines, left_width, left_middle = TreeNode.build_display(node.left)
        first_line = " " * (left_middle + 1) + "_" * (left_width - left_middle - 1) + text
        second_line = " " * left_middle + "/" + " " * (left_width - left_middle - 1 + text_width)
        shifted_left = [line + " " * text_width for line in left_lines]
        return [first_line, second_line] + shifted_left, left_width + text_width, left_width + text_width // 2

      if node.left is None:
        right_lines, right_width, right_middle = TreeNode.build_display(node.right)
        first_line = text + "_" * right_middle + " " * (right_width - right_middle)
        second_line = " " * (text_width + right_middle) + "\\" + " " * (right_width - right_middle - 1)
        shifted_right = [" " * text_width + line for line in right_lines]
        return [first_line, second_line] + shifted_right, text_width + right_width, text_width // 2

      left_lines, left_width, left_middle = TreeNode.build_display(node.left)
      right_lines, right_width, right_middle = TreeNode.build_display(node.right)
      first_line = (
        " " * (left_middle + 1)
        + "_" * (left_width - left_middle - 1)
        + text
        + "_" * right_middle
        + " " * (right_width - right_middle)
      )
      second_line = (
        " " * left_middle
        + "/"
        + " " * (left_width - left_middle - 1 + text_width + right_middle)
        + "\\"
        + " " * (right_width - right_middle - 1)
      )

      height = max(len(left_lines), len(right_lines))
      left_lines += [" " * left_width] * (height - len(left_lines))
      right_lines += [" " * right_width] * (height - len(right_lines))
      merged_lines = [left + " " * text_width + right for left, right in zip(left_lines, right_lines)]
      return [first_line, second_line] + merged_lines, left_width + text_width + right_width, left_width + text_width // 2

  @staticmethod
  def print_tree(root):
    if root is None:
      print("<empty>")
      return
    
    lines, _, _ = TreeNode.build_display(root)
    for line in lines:
      print(line.rstrip())

  @staticmethod
  def lowest_common_ancestor(root, a, b):
    if root is None:
      return None
    if root == a or root == b:
      return root

    left = TreeNode.lowest_common_ancestor(root.left, a, b)
    right = TreeNode.lowest_common_ancestor(root.right, a, b)

    if left and right:
      return root

    return left if left else right
  
  def __str__(self) -> str:
    lines, _, _ = TreeNode.build_display(self)
    return "\n".join(lines)