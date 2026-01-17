import torch
import torch.nn as nn
from typing import List, Dict, Any
from lambeq import Symbol

class EinsumModel(nn.Module):
    def __init__(self, symbols: List[Symbol] = [], sizes: List[tuple[int, ...]] = []):
        """
        symbols: a list of strings (can be any words, with punctuation, etc.)
        """
        if len(symbols) != len(sizes):
            raise ValueError("Symbols and sizes must have the same length.")

        if len(set(symbols)) != len(symbols):
            raise ValueError("Symbols must be unique.")

        super().__init__()
        self.symbols = list(symbols)
        self.sizes = list(sizes)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(size)) for size in sizes])

        self.reset_parameters()
        self.sym2weight = self.compute_sym2weight()
    
    def compute_sym2weight(self) -> Dict[Symbol, nn.Parameter]:
        """
        Compute a dictionary mapping symbols to their corresponding weights.
        This is useful for accessing the weights by symbol.
        """
        return {sym: weight for sym, weight in zip(self.symbols, self.weights)}

    def reset_parameters(self, symbols: List[Symbol] = None):
        """
        Initialize all parameters with a uniform distribution.
        Args:
            symbols: if provided, only reset the parameters for these symbols.
                     If None, reset all parameters.
        """
        def mean(size: int) -> float:
            if size < 6:
                correction_factor = [0, 3, 2.6, 2, 1.6, 1.3][size]
            else:
                correction_factor = 1 / (0.16 * size - 0.04)
            return (size/3 - 1/(15 - correction_factor)) ** 0.5

        for sym, weight in zip(self.symbols, self.weights):
            if symbols is not None and sym not in symbols:
                continue
            bound = 1 / mean(sym.directed_cod)
            nn.init.uniform_(weight, -bound, bound)

    def set_weights(self, symbols: List[Symbol], tensors: List[torch.Tensor], freeze: bool = False):
        """
        Overwrite the model's parameters with the provided tensors.
        Unknown symbols will trigger an error.
        If `freeze` is True, the parameters will not be updated during training.
        """
        if len(symbols) != len(tensors):
            raise ValueError("Symbols and tensors must have the same length.")
        
        if not all(s in self.symbols for s in symbols):
            raise ValueError(f"Some symbols {set(symbols) - set(self.symbols)} are not in the model's symbols list.")
        
        sym2idx = {sym: idx for idx, sym in enumerate(self.symbols)}
        for sym, tensor in zip(symbols, tensors):
            idx = sym2idx[sym]
            if self.weights[idx].shape != tensor.shape:
                raise ValueError(f"Shape mismatch for symbol '{sym}': "
                                    f"expected {self.weights[idx].shape}, got {tensor.shape}")
            with torch.no_grad():
                self.weights[idx].data.copy_(tensor.data)
    
    def add_symbols(self, symbols: List[Symbol], sizes: List[tuple[int, ...]]):
        """
        Add new symbols and their corresponding parameters to the model.
        If a symbol already exists, it will be ignored.
        """
        if len(symbols) != len(sizes):
            raise ValueError("Symbols and sizes must have the same length.")
        
        if any(sym in self.symbols for sym in symbols):
            raise ValueError(f"Some symbols {set(symbols) & set(self.symbols)} already exist in the model.")
        
        for sym, size in zip(symbols, sizes):
            if sym not in self.symbols:
                new_weight = nn.Parameter(torch.empty(size))
                self.symbols.append(sym)
                self.weights.append(new_weight)
                self.sizes.append(size)
        
        self.reset_parameters(symbols=symbols)
        self.sym2weight = self.compute_sym2weight()
    
    def remove_symbols(self, symbols: List[Symbol]):
        """
        Remove the specified symbols and their corresponding parameters from the model.
        """
        sym2idx = {sym: idx for idx, sym in enumerate(self.symbols)}
        indices_to_remove = [sym2idx[sym] for sym in symbols if sym in sym2idx]
        
        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            del self.symbols[idx]
            del self.weights[idx]
            del self.sizes[idx]
        
        self.sym2weight = self.compute_sym2weight()

    def _forward_single(self, input: tuple[str, List[Symbol]]) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        from cotengra import einsum
        einsum_expr, symbols = input

        return einsum(einsum_expr, *[self.sym2weight[sym] for sym in symbols])

    def forward(self, inputs: List[tuple[str, List[Symbol]]]) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            inputs: A list of tuples, where each tuple contains an einsum expression
                    and a list of symbols.
        Returns:
            A tensor representing the result of the einsum operation for each input.
        """
        return torch.stack([self._forward_single(input) for input in inputs])

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        base = super().state_dict(*args, **kwargs)
        base["symbols_list"] = self.symbols
        base["sizes_list"] = self.sizes
        return base

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if "symbols_list" in state_dict:
            loaded_symbols = state_dict.pop("symbols_list")
            self.symbols = list(loaded_symbols)
        if "sizes_list" in state_dict:
            loaded_sizes = state_dict.pop("sizes_list")
            self.sizes = list(loaded_sizes)

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(size)) for size in self.sizes]
        )
        
        self.sym2weight = self.compute_sym2weight()
        return super().load_state_dict(state_dict, strict=strict)
