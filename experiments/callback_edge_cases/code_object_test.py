# pyright: reportUnusedFunction=false
"""
Investigation of code object behavior with functools.wraps.
"""

import functools
import inspect
import sys

sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint

print("="*70)
print("CODE OBJECT INVESTIGATION")
print("="*70)

# The key question: Does __code__ get copied by @functools.wraps?

def original():
    return 1

@functools.wraps(original)
def wrapper():
    return 999

print(f"original.__code__: {original.__code__}")
print(f"wrapper.__code__: {wrapper.__code__}")
print(f"Same __code__: {original.__code__ is wrapper.__code__}")

# Wait, they're the SAME code object?
print(f"\noriginal.__code__.co_code: {original.__code__.co_code}")
print(f"wrapper.__code__.co_code: {wrapper.__code__.co_code}")

# Let me check co_consts which contains return values
print(f"\noriginal.__code__.co_consts: {original.__code__.co_consts}")
print(f"wrapper.__code__.co_consts: {wrapper.__code__.co_consts}")

# AH HA! The co_consts ARE different - (None, 1) vs (None, 999)

print("""
DISCOVERY:
@functools.wraps does NOT copy __code__!
- __code__ objects are different (different co_consts)
- But inspect.getsource() follows __wrapped__ and returns original source
- This is why hash_function_ast sees the original source

Let me verify the flow in fingerprint.hash_function_ast...
""")

# Test what hash_function_ast actually returns
h_original = fingerprint.hash_function_ast(original)
h_wrapper = fingerprint.hash_function_ast(wrapper)
print(f"hash(original): {h_original}")
print(f"hash(wrapper): {h_wrapper}")

# Now let's trace through the function:
print("\n--- Tracing hash_function_ast for wrapper ---")

try:
    source = inspect.getsource(wrapper)
    print(f"getsource returned:\n{source}")
    print("SUCCESS: Got source from getsource")
except (OSError, TypeError) as e:
    print(f"FAILED: getsource raised {e}")
    # Falls through to marshal code object path

# So getsource succeeds and returns the ORIGINAL function's source!
# That's why the hashes are equal.

print("\n" + "="*70)
print("THE ROOT CAUSE")
print("="*70)
print("""
inspect.getsource(wrapper) returns original's source because:
1. @functools.wraps sets wrapper.__wrapped__ = original
2. inspect.unwrap() is called by getsource to follow __wrapped__
3. The source of the ORIGINAL function is returned

This means the actual wrapper code is NEVER seen by hash_function_ast.

Proof:
""")

# Let's verify with unwrap
unwrapped = inspect.unwrap(wrapper)
print(f"inspect.unwrap(wrapper) is original: {unwrapped is original}")

# What about nested wrappers?
print("\n--- Testing nested wrappers ---")

def level0():
    return 0

@functools.wraps(level0)
def level1():
    return 1

@functools.wraps(level1)
def level2():
    return 2

print(f"level0 source:\n{inspect.getsource(level0)}")
print(f"level1 source:\n{inspect.getsource(level1)}")
print(f"level2 source:\n{inspect.getsource(level2)}")

# All three return level0's source!

h0 = fingerprint.hash_function_ast(level0)
h1 = fingerprint.hash_function_ast(level1)
h2 = fingerprint.hash_function_ast(level2)
print(f"\nhash(level0): {h0}")
print(f"hash(level1): {h1}")
print(f"hash(level2): {h2}")

print("""
CONFIRMED: All three functions have the SAME hash because they all
trace back to level0's source through the __wrapped__ chain.

This is a CRITICAL vulnerability:
- Multiple layers of decoration are invisible
- Each layer could add significant behavior changes
- None of it is fingerprinted
""")

# Test the marshal fallback
print("\n--- Testing marshal fallback ---")

# If we could prevent getsource from working, would marshal work?
# Create a lambda (which doesn't have reliable source)
lambda_func = lambda x: x * 2  # noqa: E731
h_lambda = fingerprint.hash_function_ast(lambda_func)
print(f"Lambda hash (should use marshal): {h_lambda}")

# For a wrapped function, getsource DOES work (but returns wrong source)
# So marshal is never tried

print("\n--- Potential fix verification ---")

# What if we checked for __wrapped__ and used __code__ instead?
import marshal
import xxhash

def hash_by_code(func):
    """Hash using __code__ directly (not getsource)."""
    return xxhash.xxh64(marshal.dumps(func.__code__)).hexdigest()

print(f"original by code: {hash_by_code(original)}")
print(f"wrapper by code: {hash_by_code(wrapper)}")
print(f"Different: {hash_by_code(original) != hash_by_code(wrapper)}")

print("""
FIX: Using marshal.dumps(__code__) would correctly distinguish
original from wrapper because __code__ is NOT copied by @wraps.

However, this loses the benefit of ignoring whitespace/docstrings.
A better fix might be to:
1. Detect __wrapped__ attribute
2. Hash BOTH the wrapper (via __code__) AND the wrapped (via AST)
3. Combine the hashes
""")
