#!/usr/bin/env python3
"""Test script to check Python version and type annotation support."""

import sys
from typing import Union

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test type annotations
def test_old_style(x: Union[str, int]) -> str:
    return str(x)

def test_new_style(x: str | int) -> str:  # This will fail on Python < 3.10
    return str(x)

print("Type annotation test passed!")
print("Both old and new style type annotations work.")
