import numpy as np
import pytest

try:
    from einops import rearrange as original_rearrange
except ImportError:
    pytest.skip("Original einops library not found, skipping compatibility tests", allow_module_level=True)

from einops_impl.rearrange import rearrange as your_rearrange


@pytest.mark.parametrize("shape, pattern, named_axes", [
    # Simple cases that original einops supports
    ((30, 40), 'h w -> w h', {}),
    ((30, 40), '(h1 h2) w -> h1 h2 w', {'h1': 5}),
    ((2, 3, 4, 5), '... h w -> ... (h w)', {}),
    ((1, 28, 28, 3), 'b h w c -> b c h w', {}),
    ((10, 32, 32), 'b (h h1) (w w1) -> b (h w) (h1 w1)', {'h1': 4, 'w1': 4}),
    ((8, 16, 16, 3), 'b (h p1) (w p2) c -> b h w (p1 p2 c)', {'p1': 4, 'p2': 4}),
])
def test_against_original(shape, pattern, named_axes):
    """Testing einops implementation against original einops library."""
    np.random.seed(42)
    x = np.random.rand(*shape)
    
    result_original = original_rearrange(x, pattern, **named_axes)
    result_yours = your_rearrange(x, pattern, **named_axes)
    
    assert result_original.shape == result_yours.shape
    assert np.allclose(result_original, result_yours)
