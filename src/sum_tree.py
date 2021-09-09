from typing import Collection, List, Tuple

import numpy as np


class Node:
    def __init__(self, left: 'Node' = None, right: 'Node' = None, idx=None, is_leaf=False):
        self.value = None
        self.left_child = left
        self.right_child = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left_child.value + self.right_child.value
        self.parent = None
        self.idx = idx
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self
    
    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, idx=idx, is_leaf=True)
        leaf.value = value
        return leaf
    
    def retrieve(self, value):
        if self.is_leaf:
            return self
        
        if self.left_child.value >= value:
            return self.left_child.retrieve(value)
        else:
            return self.right_child.retrieve(value - self.left_child.value)
    
    def update(self, new_value):
        change = new_value - self.value
        self.value = new_value
        self.parent.propagate(change)
    
    def propagate(self, change):
        self.value += change
        if self.parent:
            self.parent.propagate(change)
    
    def bulk_retrieve(self, values, output_list: List, idx_list: List):
        if self.is_leaf:
            output_list.extend([self.value] * len(values))
            idx_list.extend([self.idx] * len(values))
        else:
            left_values = values[self.left_child.value >= values]
            right_values = values[self.left_child.value < values] - self.left_child.value
            if np.any(left_values):
                self.left_child.bulk_retrieve(left_values, output_list, idx_list)
            if np.any(right_values):
                self.right_child.bulk_retrieve(right_values, output_list, idx_list)


class SumTree:
    """ Based on the following implementation:
    https://adventuresinmachinelearning.com/sumtree-introduction-python/"""
    
    def __init__(self, tree_inputs: Collection, seed: int):
        nodes = [Node.create_leaf(inp, idx) for idx, inp in enumerate(tree_inputs)]
        self.leafs = nodes
        while len(nodes) > 1:
            inodes = iter(nodes)
            nodes = [Node(*pair) for pair in zip(inodes, inodes)]
        self.top_node = nodes[0]
        self.max = max(tree_inputs)
        self.rng = np.random.Generator(np.random.PCG64(seed))
    
    def sample(self, num_samples) -> Tuple[List, List]:
        values = self.rng.uniform(0, self.top_node.value, num_samples)
        return self.bulk_retrieve(values)
    
    def retrieve(self, value):
        return self.top_node.retrieve(value)
    
    def bulk_retrieve(self, values: Collection) -> Tuple[List, List]:
        output_list = []
        idx_list = []
        self.top_node.bulk_retrieve(values, output_list, idx_list)
        return output_list, idx_list
    
    def update(self, value: float, idx: int):
        self.leafs[idx].update(float(value))
        self.max = max(value, self.max)
    
    def bulk_update(self, values: Collection, idxs: List[int]):
        for val, idx in zip(values, idxs):
            self.leafs[idx].update(val.detach().item())
        self.max = max(max(values), self.max)
