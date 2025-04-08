# EinOps Implementation

This project implements a custom version of the popular [einops](https://github.com/arogozhnikov/einops) library for tensor manipulation. Currently it just supports rearrange function with added support to nested grouping (which is not available in einops)

## Colab notebook link 
https://colab.research.google.com/drive/1Pnq9cm2I4b6v-4x0mSMhFRLTIkARAFQH?usp=sharing

## Features

- **Standard einops operations**: Support for tensor reshaping and transposition using the familiar einops pattern syntax
- **Enhanced nested pattern support**: Unlike the original einops, our implementation supports multi-level nested parentheses, allowing for more complex tensor transformations
- **Comprehensive test suite**: Unit tests and compatibility tests to ensure correctness

## Design Decisions

### Parser

The core of our implementation is the pattern parser (`einops_impl/parser.py`), which:
- Parses the einops-style pattern strings
- Handles nested groups with parentheses (including multi-level nesting)
- Identifies and validates axes and dimensions

### Shape Analysis

The shape analyzer (`einops_impl/shape_analzer.py`):
- Determines axis sizes from input tensors
- Handles dynamic resolution of unknown dimensions
- Validates that the reshaping operation is valid

### Operations

Our operations module (`einops_impl/operations.py`) provides:
- Core tensor manipulation operations (split, merge, transpose, expand)
- Clean interfaces for implementing higher-level operations

### Rearrangement

The main `rearrange` function:
- Uses the components above to transform tensors according to the provided pattern
- Flattens complex nested patterns into a series of simple operations
- Maintains dimension consistency throughout transformations

## Installation
## Installation

Clone the repository

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from einops_impl.rearrange import rearrange

# Basic usage - same as original einops
x = np.random.rand(30, 3)
result = rearrange(x, '(h w) c -> h w c', h=5)
# result.shape == (5, 6, 3)

# Enhanced nested pattern support
x = np.random.rand(24, 5)
result = rearrange(x, '((a b) c) d -> a (b (c d))', a=2, b=3, c=4)
# result.shape == (2, 60)
```

## Running Tests

To run all tests:

```bash
pytest
```

To run specific test files:

```bash
# Run compatibility tests
pytest einops_impl/tests/test_compatibility.py

# Run parser tests
pytest einops_impl/tests/test_parser.py

# Run rearrange tests
pytest einops_impl/tests/test_rearrange.py
```
