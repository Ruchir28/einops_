from typing import Dict, Any
import numpy as np
from .parser import Parser
from .shape_analzer import ShapeAnalyzer
from .operations import Operations


def expand_group(group_name, grouped_axes):
    expansion = []
    for item in grouped_axes[group_name]:
        if item in grouped_axes:
            expansion.extend(expand_group(item, grouped_axes))
        else:
            expansion.append(item)
    return expansion

def rearrange(tensor: np.ndarray, pattern: str, **axis_lengths) -> np.ndarray:
    """
    Rearrange a tensor according to the given pattern.
    
    Args:
        tensor: Input numpy array
        pattern: Einops-style pattern string
        **axis_lengths: Known axis lengths
    
    Returns:
        Rearranged numpy array
    
    Example:
        >>> x = np.random.rand(30, 3)
        >>> result = rearrange(x, '(h w) c -> h w c', h=5)
        >>> result.shape
        (5, 6, 3)
    """
    # 1. Parse the pattern
    parser = Parser(pattern)
    input_axes, output_axes = parser.parse()

    
    # 2. Analyze shapes
    shape_analyzer = ShapeAnalyzer()
    axis_sizes = shape_analyzer.get_axis_size(
        tensor, input_axes, parser.grouped_axes, axis_lengths
    )
    
    # 3. Initialize operations
    ops = Operations()
    current = tensor
    
    # 4. Process input grouping
    input_composition = []  # Track how axes are composed
    curr_original_idx = 0
    for i, axis in enumerate(input_axes):
        if axis == '...':
            # Handle ellipsis
            ellipsis_dims = len(axis_sizes['...'])
            input_composition.extend([f'...{i}' for i in range(ellipsis_dims)])
            curr_original_idx += ellipsis_dims
        elif axis in parser.grouped_axes:
            # Split grouped axes
            group_axes = expand_group(axis, parser.grouped_axes)
            sizes = tuple(axis_sizes[ax] for ax in group_axes)
            current = ops.split_axis(current, curr_original_idx, sizes)
            input_composition.extend(group_axes)
            curr_original_idx += len(group_axes) # because we are splitting the axis
        else:
            input_composition.append(axis)
            curr_original_idx += 1
    
    # 5. Plan output composition
    output_composition = []
    for axis in output_axes:
        if axis == '...':
            output_composition.extend([f'...{i}' for i in range(len(axis_sizes['...']))])
        elif axis in parser.grouped_axes:
            output_composition.extend(expand_group(axis, parser.grouped_axes))
        else:
            output_composition.append(axis)

    if len(input_composition) != len(output_composition):
        raise ValueError(f"Inconsistent number of dimensions")
        
    # process them for expansion
    for i, (in_axis, out_axis) in enumerate(zip(input_composition, output_composition)):
        # Skip ellipsis markers
        if in_axis.startswith('...'):
            continue
            
        # Check for dimension changes
        if in_axis == '1':
            out_size = axis_lengths[out_axis]
            current = ops.expand_axis(current, i, out_size)
            input_composition[i] = out_axis # update the input composition's axis which is 1 to the variable used in the output so that it works for permutation
    
    # 6. Create permutation for transpose
    perm = [input_composition.index(ax) for ax in output_composition]
    current = ops.transpose_axes(current, perm)
    
    # 7. Process output grouping
    final_shape = []
    i = 0
    for axis in output_axes:
        if axis == '...':
            final_shape.extend(axis_sizes['...'])
            i += len(axis_sizes['...'])
        elif axis in parser.grouped_axes:
            size = np.prod([axis_sizes[ax] for ax in expand_group(axis, parser.grouped_axes)])
            final_shape.append(size)
            i += len(expand_group(axis, parser.grouped_axes))
        else:
            final_shape.append(axis_sizes[axis])
            i += 1
    
    return current.reshape(final_shape)