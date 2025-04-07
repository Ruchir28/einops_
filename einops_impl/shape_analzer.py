import numpy as np
from typing import Dict,List,Tuple,Union
class ShapeAnalyzer:
    @staticmethod
    def get_axis_size(tensor: np.ndarray, axes: List[str],
                      grouped_axes: Dict[str,List[str]],
                      axis_lengths: Dict[str,int]) -> Dict[str,int]:
        """
        Get the size of the axes in the tensor
        """
        sizes = {}
        current_shape = tensor.shape

        for axis in axis_lengths:
            sizes[axis] = axis_lengths[axis]

        def calculate_group_size(group_axes: List[str]) -> int:
            total_size = 1
            for axis in group_axes:
                if axis in axis_lengths:
                    total_size *= axis_lengths[axis]
                elif axis in sizes:
                    total_size *= sizes[axis]
                elif axis in grouped_axes:
                    total_size *= calculate_group_size(grouped_axes[axis])
                else:
                    raise ValueError(f"Cannot determine size for axis {axis}")
            return total_size

        shape_idx = 0

        for axis in axes:
            if axis == '...':
                remaining_dims = len(current_shape) - (len(axes) - 1) # it's allowed only once
                if remaining_dims < 0:
                    raise ValueError("Not enough dimensions")
                sizes['...'] = current_shape[shape_idx:shape_idx+remaining_dims]
                shape_idx += remaining_dims
            elif axis in grouped_axes:
                # group axis
                group_size = current_shape[shape_idx]
                group_axes = grouped_axes[axis]
                
                # Calculate known product for the group
                known_product = 1
                unknown_axis = None
                for group_axis in group_axes:
                    if group_axis in sizes or group_axis in axis_lengths:
                        known_product *= (sizes.get(group_axis) or axis_lengths[group_axis])
                    else:
                        if unknown_axis is not None:
                            raise ValueError(f"Multiple unknown axes in group: {group_axes}")
                        unknown_axis = group_axis
                
                # If we have one unknown axis, we can infer its size
                if unknown_axis is not None:
                    sizes[unknown_axis] = group_size // known_product
                    if group_size % known_product != 0:
                        raise ValueError(
                            f"Cannot evenly divide group size {group_size} by known product {known_product}"
                        )

                 
                expected_size = calculate_group_size(grouped_axes[axis])
                if group_size != expected_size:
                    raise ValueError(f"Group size mismatch for axis {axis}. "
                                     f"Expected {expected_size}, got {group_size}")
                shape_idx += 1
            else:
                if axis not in sizes:
                    sizes[axis] = current_shape[shape_idx]
                shape_idx += 1

        return sizes


        