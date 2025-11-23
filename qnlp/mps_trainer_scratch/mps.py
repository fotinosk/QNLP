"""
This is not a generic MPS implementaion, it is a specific one following the paper
Supervised Learning with Quantum Insipred Tensor Networks specifically for MNIST
"""

import numpy as np

class MPS:
    def __init__(
        self,
        num_labels: int,
        bond_dim: int,
        input_data_size: int,
        batch_size: int
    ) -> None:
        pass
    
    def move_label_index(self, site: int):
        pass
    
    def forward(self, x):
        pass
    
    def backprop(self, x, label):
        # this is better outside this function but makes it easier for now
        pass
    
    @staticmethod
    def _embed_input(x):
        # this is better outside this function but makes it easier for now
        pass
    
    @staticmethod
    def _preprocess_input(x):
        pass