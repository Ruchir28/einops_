from typing import Dict, Any
import numpy as np
from .parser import Parser
from .shape_analzer import ShapeAnalyzer
from .operations import Operations


def expand_group(group_name, grouped_axes):
    if group_name not in grouped_axes:
        raise ValueError(f"Group '{group_name}' not found in the pattern. Available groups: {list(grouped_axes.keys())}")
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
    
    Raises:
        ValueError: If tensor is None or not a numpy array
        ValueError: If pattern is empty or None
        ValueError: If axis lengths are missing or invalid
    """
    # Input validation
    if tensor is None:
        raise ValueError("Input tensor cannot be None")
    if not isinstance(tensor, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(tensor).__name__}")
    if not pattern:
        raise ValueError("Pattern string cannot be empty")
    
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
        raise ValueError(f"Inconsistent number of dimensions: expected {len(output_composition)}, got {len(input_composition)}")
        
    # process them for expansion
    for i, (in_axis, out_axis) in enumerate(zip(input_composition, output_composition)):
        # Skip ellipsis markers
        if in_axis.startswith('...'):
            continue
            
        # Check for dimension changes
        if in_axis == '1':
            if out_axis not in axis_lengths:
                raise ValueError(f"Missing size for expansion axis '{out_axis}'. Please provide it in axis_lengths.")
            out_size = axis_lengths[out_axis]
            if out_size <= 0:
                raise ValueError(f"Expansion size for axis '{out_axis}' must be positive, got {out_size}")
            current = ops.expand_axis(current, i, out_size)
            input_composition[i] = out_axis # update the input composition's axis which is 1 to the variable used in the output so that it works for permutation
    
    # 6. Create permutation for transpose
    try:
        perm = [input_composition.index(ax) for ax in output_composition]
    except ValueError as e:
        raise ValueError(f"Invalid axis in output pattern. This might be due to mismatched axes between input and output patterns.") from e
    current = ops.transpose_axes(current, perm)
    
    # 7. Process output grouping
    final_shape = []
    i = 0
    for axis in output_axes:
        if axis == '...':
            final_shape.extend(axis_sizes['...'])
            i += len(axis_sizes['...'])
        elif axis in parser.grouped_axes:
            group_axes = expand_group(axis, parser.grouped_axes)
            try:
                size = np.prod([axis_sizes[ax] for ax in group_axes])
            except KeyError as e:
                missing_axis = str(e).strip("'")
                raise ValueError(f"Missing size for grouped axis '{missing_axis}' in group {group_axes}") from e
            final_shape.append(size)
            i += len(group_axes)
        else:
            if axis not in axis_sizes:
                raise ValueError(f"Missing size for axis '{axis}'. This might be due to an undefined axis in the pattern.")
            final_shape.append(axis_sizes[axis])
            i += 1
    
    try:
        return current.reshape(final_shape)
    except ValueError as e:
        total_elements_before = np.prod(current.shape)
        total_elements_after = np.prod(final_shape)
        if total_elements_before != total_elements_after:
            raise ValueError(f"Cannot reshape tensor of size {total_elements_before} into shape {tuple(final_shape)} "
                           f"(which has size {total_elements_after}). This might be due to incorrect axis sizes.") from e
        raise  