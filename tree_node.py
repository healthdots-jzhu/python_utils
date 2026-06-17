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
  
  @staticmethod
  def find_node_by_value(root, value):
    if (root.value == value):
      return root

    left = TreeNode.find_node_by_value(root.left, value) if root.left else None
    if (left is not None):
      return left

    right = TreeNode.find_node_by_value(root.right, value) if root.right else None
    if (right is not None):
      return right

    return None

if __name__ == "__main__":
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
  b = TreeNode.find_node_by_value(root, 9)
  print(root)
  print(TreeNode.lowest_common_ancestor(root, a, b).value)