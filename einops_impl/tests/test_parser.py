import pytest
from einops_impl.parser import Parser


def test_simple_pattern():
    parser = Parser('h w -> w h')
    input_axes, output_axes = parser.parse()
    assert input_axes == ['h', 'w']
    assert output_axes == ['w', 'h']
    assert parser.axes_names == {'h', 'w'}
    assert parser.grouped_axes == {}

def test_basic_grouping():
    parser = Parser('(h w) c -> h w c')
    input_axes, output_axes = parser.parse()
    assert input_axes == ['group_0', 'c']
    assert output_axes == ['h', 'w', 'c']
    assert parser.grouped_axes['group_0'] == ['h', 'w']
    assert parser.axes_names == {'h', 'w', 'c'}

def test_ellipsis():
    parser = Parser('... h w -> ... (h w)')
    input_axes, output_axes = parser.parse()
    assert input_axes == ['...', 'h', 'w']
    assert output_axes == ['...', 'group_0']
    assert parser.grouped_axes['group_0'] == ['h', 'w']

def test_nested_groups():
    parser = Parser('((h w) d) c -> h w (d c)')
    input_axes, output_axes = parser.parse()
    assert 'group_0' in parser.grouped_axes  # (h w)
    assert 'group_1' in parser.grouped_axes  # ((h w) d)
    assert 'group_2' in parser.grouped_axes  # (d c)
    assert parser.grouped_axes['group_0'] == ['h', 'w']
    assert parser.grouped_axes['group_1'] == ['group_0', 'd']
    assert parser.grouped_axes['group_2'] == ['d', 'c']


def test_multiple_groups():
    parser = Parser('(h w) (d c) -> (h d) (w c)')
    input_axes, output_axes = parser.parse()
    assert len(parser.grouped_axes) == 4
    assert set(parser.axes_names) == {'h', 'w', 'd', 'c'}

# Space Handling Tests
def test_various_spacing():
    patterns = [
        'h w->w h',
        'h w   ->   w h',
        'h    w -> w    h',
        '(h w)->  (w h)'
    ]
    for pattern in patterns:
        parser = Parser(pattern)
        input_axes, output_axes = parser.parse()
        assert input_axes == ['h', 'w'] or input_axes == ['group_0']

def test_unmatched_parentheses():
    invalid_patterns = [
        '(h w -> h w',
        'h w) -> h w',
        '((h w) -> h w',
    ]
    for pattern in invalid_patterns:
        with pytest.raises(ValueError):
            Parser(pattern).parse()

def test_missing_arrow():
    with pytest.raises(ValueError):
        Parser('h w w h').parse()

# def test_empty_groups():
#     with pytest.raises(ValueError):
#         Parser('() h -> h').parse()

# Advanced Pattern Tests
def test_complex_patterns():
    patterns = [
        '(h w) c d -> h w (c d)',
        '... (h w) -> ... h w',
        '(h1 w1) (h2 w2) -> (h1 h2) (w1 w2)',
        'b (h w) c -> b h w c'
    ]
    for pattern in patterns:
        parser = Parser(pattern)
        input_axes, output_axes = parser.parse()
        assert isinstance(input_axes, list)
        assert isinstance(output_axes, list)
