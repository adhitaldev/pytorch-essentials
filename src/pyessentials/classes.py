""""
Demonstrates various usage of classes, their functionalities, 
and usage, namely sorting, iterator, etc.
"""

class Node:
    def __init__(self, val):
        self.val = val

    # Defines less than for sorting
    def __lt__(self, other):
        return self.val < other.val

    # Defines larger than for sorting
    def __rt__(self, other):
        return self.val > other.val
    
    def __eq__(self, other):
        """
        Special case where they are equal if the other value is twice that of the current one.
        """
        return isinstance(other, Node) and self.val  == other.val // 2
    
    def __hash__ (self, other):
        return self.__eq__()
    
class NodeIterator:
    def __init__(self, node_list):
        self.node_list = node_list
        self.idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx < len(self.node_list):
            item = self.node_list[self.idx]
            self.idx += 1
            return item.val
        else:
            raise StopIteration

def run_examples():
    """
    Runs various examples of classes and its functionalities
    """
    n1 = Node(20)
    n2 = Node(10)
    n3 = Node(1)
    nodes = [n1, n2, n3]
    sorted_nodes = sorted(nodes)
    print(f"Before sorting: {[node.val for node in nodes]}")
    print(f"After sorting: {[node.val for node in sorted_nodes]}")

    # Iteration
    print('Iterator')
    node_iter = NodeIterator(nodes)
    for n in node_iter:
        print(n)
    node_iter = NodeIterator(sorted_nodes)
    for n in node_iter:
        print(n)
    
    print('Checking equality ')
    print(Node(10) == Node(10)) # should return false
    print(Node(10) == Node(20)) # should return true
if __name__== "__main__":
    run_examples()