import numpy as np
from typing import List, Tuple

class Operations:
    @staticmethod
    def split_axis(tensor: np.ndarray, axis: int, sizes: Tuple[int, ...]) -> np.ndarray:
        """
        Split an axis into multiple axes.
        
        Example:
            tensor.shape = (30, 3)
            split_axis(tensor, axis=0, sizes=(5, 6))
            -> result.shape = (5, 6, 3)
        """
        shape = list(tensor.shape)
        if np.prod(sizes) != shape[axis]:
            raise ValueError(
                f"Cannot split axis {axis} of size {shape[axis]} into {sizes}. "
                f"Product of sizes {np.prod(sizes)} does not match axis size."
            )
        new_shape = shape[:axis] + list(sizes) + shape[axis + 1:]
        return tensor.reshape(new_shape)

    @staticmethod
    def merge_axes(tensor: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
        """
        Merge multiple consecutive axes into one.
        
        Example:
            tensor.shape = (5, 6, 3)
            merge_axes(tensor, axes=(0, 1))
            -> result.shape = (30, 3)
        """
        shape = list(tensor.shape)
        new_size = np.prod([shape[i] for i in axes])
        new_shape = (
            shape[:axes[0]] + 
            [new_size] + 
            shape[axes[-1] + 1:]
        )
        return tensor.reshape(new_shape)

    @staticmethod
    def transpose_axes(tensor: np.ndarray, axes: List[int]) -> np.ndarray:
        """
        Reorder axes according to permutation.
        
        Example:
            tensor.shape = (5, 6, 3)
            transpose_axes(tensor, axes=[2, 0, 1])
            -> result.shape = (3, 5, 6)
        """
        return np.transpose(tensor, axes)
    
    @staticmethod
    def expand_axis(tensor: np.ndarray, axis: int, size: int) -> np.ndarray:
        """Expand a size-1 dimension to target size"""
        if tensor.shape[axis] != 1:
            raise ValueError(f"Can only expand axes of size 1, got {tensor.shape[axis]}")
        return np.repeat(tensor, size, axis=axis)
    
