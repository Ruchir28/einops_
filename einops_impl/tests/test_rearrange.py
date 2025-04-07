import pytest
import numpy as np
from einops_impl.rearrange import rearrange

def assert_shapes_equal(actual, expected):
    assert actual == expected, f"Shape mismatch. Expected {expected}, got {actual}"

def test_basic_ellipsis():
    """Test basic ellipsis handling with simple shape transformations"""
    # Test case 1: Preserve batch dimensions
    x = np.random.rand(2, 3, 30, 40)
    result = rearrange(x, '... h w -> ... (h w)')
    assert_shapes_equal(result.shape, (2, 3, 1200))
    
    # Test case 2: Single batch dimension
    # x = np.random.rand(2, 30, 40)
    # result = rearrange(x, '... h w -> ... (h w)')
    # assert_shapes_equal(result.shape, (2, 1200))

def test_multiple_batch_dimensions():
    """Test handling of multiple batch dimensions"""
    # Test case 1: Three batch dimensions
    x = np.random.rand(2, 3, 4, 30, 40)
    result = rearrange(x, '... h w -> ... (h w)')
    assert_shapes_equal(result.shape, (2, 3, 4, 1200))
    
    # Test case 2: Batch with grouping
    x = np.random.rand(2, 3, 4, 32, 32)
    result = rearrange(x, '... (h ph) (w pw) -> ... h w (ph pw)', 
                        ph=8, pw=8)
    assert_shapes_equal(result.shape, (2, 3, 4, 4, 4, 64))


def test_ellipsis_with_grouping():
    """Test combination of ellipsis and grouped dimensions"""
    # Test case 1: Group after ellipsis
    x = np.random.rand(2, 3, 30, 40)
    result = rearrange(x, '... (h1 w1) (h2 w2) -> ... (h1 h2) (w1 w2)', 
                        h1=5, h2=5)
    assert_shapes_equal(result.shape, (2, 3, 25, 48))
    
    # Test case 2: Complex grouping with ellipsis
    x = np.random.rand(2, 3, 32, 32)
    result = rearrange(x, '... (h ph) (w pw) -> ... h w (ph pw)', 
                        ph=8, pw=8)
    assert_shapes_equal(result.shape, (2, 3, 4, 4, 64))

def test_ellipsis_with_reordering():
    """Test reordering dimensions with ellipsis"""
    # Test case 1: Reorder after ellipsis
    x = np.random.rand(2, 3, 30, 40)
    result = rearrange(x, '... h w -> ... w h')
    assert_shapes_equal(result.shape, (2, 3, 40, 30))
    
    # Test case 2: Complex reordering
    x = np.random.rand(2, 3, 4, 32, 32)
    result = rearrange(x, '... c (h ph) (w pw) -> ... (h w) c (ph pw)', 
                        ph=8, pw=8)
    assert_shapes_equal(result.shape, (2, 3, 16, 4, 64))

def test_ellipsis_validation():
    """Test validation of ellipsis usage"""
    x = np.random.rand(2, 3, 30, 40)
    
    # Test case 1: Multiple ellipsis
    with pytest.raises(ValueError, match="Pattern must contain at most one ellipsis"):
        rearrange(x, '... h ... w -> ... (h w)')
    
    # Test case 2: Inconsistent ellipsis
    with pytest.raises(ValueError, match="Inconsistent number of dimensions"):
        rearrange(x, '... h w -> ... h')


def test_shape_inference():
    """Test shape inference with ellipsis"""
    # Test case 1: Infer dimensions with ellipsis
    x = np.random.rand(2, 3, 32, 32)
    result = rearrange(x, '... (h p1) (w p2) -> ... h w (p1 p2)', 
                        p1=8, p2=8)
    assert_shapes_equal(result.shape, (2, 3, 4, 4, 64))
    
    # Test case 2: Complex inference
    x = np.random.rand(2, 3, 30, 40)
    result = rearrange(x, '... (h1 w1) (h2 w2) -> ... h1 h2 (w1 w2)', 
                        h1=5, h2=5)
    assert_shapes_equal(result.shape, (2, 3, 5, 5, 48))

def test_edge_cases():
    """Test edge cases and special situations"""
    # Test case 1: Single dimension ellipsis
    x = np.random.rand(2, 30, 40)
    result = rearrange(x, '... h w -> ... (h w)')
    assert_shapes_equal(result.shape, (2, 1200))
    
    # Test case 2: Empty batch dimensions
    x = np.random.rand(30, 40)
    result = rearrange(x, '... h w -> ... (h w)')
    assert_shapes_equal(result.shape, (1200,))
    
    # Test case 3: Single dimension after ellipsis
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, '... d -> ... d')
    assert_shapes_equal(result.shape, (2, 3, 4))

def test_error_cases():
    """Test error handling"""
    x = np.random.rand(2, 3, 30, 40)
    
    # Test case 1: Invalid pattern
    with pytest.raises(ValueError):
        rearrange(x, '... h w -> ... h w k')
    
    # Test case 2: Missing dimensions
    with pytest.raises(ValueError):
        rearrange(x, '... h w -> ... h')
    
    # Test case 3: Invalid axis lengths
    with pytest.raises(ValueError):
        rearrange(x, '... (h p) w -> ... h p w', p=7)  # 30 not divisible by 7

def test_numeric_expansion():
    """Test numeric expansion"""
    x = np.random.rand(2, 1, 3)
    result = rearrange(x,'a 1 c -> a b c', b=3)
    assert_shapes_equal(result.shape, (2, 3, 3))
    
    