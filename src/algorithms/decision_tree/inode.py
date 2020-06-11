from algorithms.decision_tree.node import Node


class Inode(Node):
    split_index: int
    split_value: float
    child_true: Node
    child_false: Node

    def __init__(self, split_index: int, split_value: float, child_true: Node, child_false: Node):
        super().__init__()
        self.split_index = split_index
        self.split_value = split_value
        self.child_true = child_true
        self.child_false = child_false
        self.analyze_node()

    def analyze_node(self):
        self.child_true.analyze_node()
        self.child_false.analyze_node()
        self.data_count = self.child_true.data_count + self.child_false.data_count
        self.entropy = self.child_true.data_count / self.data_count * self.child_true.entropy
        self.entropy += self.child_false.data_count / self.data_count * self.child_false.entropy

    def print_node(self, level: int = 0) -> str:
        spacer = " " * 2 * level
        level += 1
        return "{}Inode: Split-Index: {}, Split-Value: {}\n" \
               "{}- True:\n" \
               "{}\n" \
               "{}- False:\n" \
               "{}"\
            .format(spacer, self.split_index, self.split_value,
                    spacer, self.child_true.print_node(level),
                    spacer, self.child_false.print_node(level))
