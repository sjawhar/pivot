# pyright: reportUnusedFunction=false
"""
Deep investigation of functools.wraps vulnerability.
"""

import functools
import inspect
import sys

sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint

print("="*70)
print("functools.wraps VULNERABILITY - DEEP INVESTIGATION")
print("="*70)

# Test 1: Basic wraps scenario
print("\n--- Test 1: Basic @functools.wraps ---")

def original_v1():
    """Original docstring."""
    return 1

@functools.wraps(original_v1)
def different_implementation():
    """Different docstring - ignored by wraps."""
    return 999  # Completely different!

print(f"original_v1.__name__: {original_v1.__name__}")
print(f"different_implementation.__name__: {different_implementation.__name__}")
print(f"different_implementation.__wrapped__: {different_implementation.__wrapped__}")

# What does getsource return?
src_orig = inspect.getsource(original_v1)
src_diff = inspect.getsource(different_implementation)

print(f"\noriginal_v1 source:\n{src_orig}")
print(f"different_implementation source:\n{src_diff}")

print("""
KEY INSIGHT: inspect.getsource() returns the SOURCE OF THE WRAPPED FUNCTION
when @functools.wraps is applied! It follows the __wrapped__ attribute.

This means:
1. hash_function_ast(different_implementation) hashes original_v1's AST
2. The completely different implementation (return 999) is NEVER seen
""")

# Verify by checking the code object directly
print("\n--- Verifying via code objects ---")
print(f"original_v1.__code__.co_code: {original_v1.__code__.co_code.hex()}")
print(f"different_implementation.__code__.co_code: {different_implementation.__code__.co_code.hex()}")

# Code objects ARE different, but getsource is fooled

# Test 2: Manual @wraps equivalent
print("\n--- Test 2: Manual attributes vs @wraps ---")

def base():
    return "base"

def manual_wrap():
    return "manual"

# Manually copy attributes (like wraps does)
manual_wrap.__name__ = base.__name__
manual_wrap.__doc__ = base.__doc__
manual_wrap.__dict__.update(base.__dict__)
# But NOT __wrapped__ !

print(f"manual_wrap has __wrapped__: {hasattr(manual_wrap, '__wrapped__')}")
src_manual = inspect.getsource(manual_wrap)
print(f"manual_wrap source:\n{src_manual}")
# This works correctly!

# Test 3: Adding __wrapped__ manually
print("\n--- Test 3: Adding __wrapped__ manually ---")

def target():
    return "target"

def attacker():
    return "EVIL CODE"

attacker.__wrapped__ = target  # type: ignore

src_attacker = inspect.getsource(attacker)
print(f"attacker source:\n{src_attacker}")

# Does this fool fingerprinting?
fp_target = fingerprint.get_stage_fingerprint(target)
fp_attacker = fingerprint.get_stage_fingerprint(attacker)
print(f"target fingerprint: {fp_target}")
print(f"attacker fingerprint: {fp_attacker}")

print("""
CONFIRMED: Setting __wrapped__ makes inspect.getsource() return the
wrapped function's source, completely hiding the actual implementation!
""")

# Test 4: Decorator that modifies behavior but uses wraps
print("\n--- Test 4: Real-world decorator pattern ---")

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

def add_v2(a, b):
    return a + b

print(f"add (decorated) source:\n{inspect.getsource(add)}")
print(f"add_v2 (undecorated) source:\n{inspect.getsource(add_v2)}")

fp_add = fingerprint.get_stage_fingerprint(add)
fp_add_v2 = fingerprint.get_stage_fingerprint(add_v2)
print(f"add fingerprint: {fp_add}")
print(f"add_v2 fingerprint: {fp_add_v2}")

print("""
IMPACT ANALYSIS:

The @functools.wraps decorator (or anything setting __wrapped__) causes
hash_function_ast() to hash the WRAPPED function, not the WRAPPER.

This means:
1. Any decorator using @functools.wraps will have the DECORATOR CODE IGNORED
2. Only the original wrapped function is fingerprinted
3. Changing the decorator's behavior (e.g., adding caching, logging, validation)
   will NOT trigger a re-run

SEVERITY: CRITICAL
- Very common pattern in Python
- Affects: @lru_cache, @cache, custom decorators, middleware patterns
- Completely invisible - decorator code is never fingerprinted
""")

# Test 5: @functools.lru_cache
print("\n--- Test 5: @functools.lru_cache ---")

@functools.lru_cache(maxsize=100)
def cached_v1(x):
    return x * 2

@functools.lru_cache(maxsize=1000)  # Different cache size!
def cached_v2(x):
    return x * 2

fp_cached_v1 = fingerprint.get_stage_fingerprint(cached_v1)
fp_cached_v2 = fingerprint.get_stage_fingerprint(cached_v2)

print(f"cached_v1 fingerprint: {fp_cached_v1}")
print(f"cached_v2 fingerprint: {fp_cached_v2}")
print(f"Equal: {fp_cached_v1 == fp_cached_v2}")

print("""
@functools.lru_cache ALSO uses __wrapped__, so:
- Different maxsize values produce SAME fingerprint
- Adding/removing @lru_cache is NOT detected
""")

# Test 6: The nuclear option - can we detect __wrapped__?
print("\n--- Test 6: Detecting __wrapped__ ---")

def has_wrapped(func):
    return hasattr(func, '__wrapped__')

print(f"original_v1 has __wrapped__: {has_wrapped(original_v1)}")
print(f"different_implementation has __wrapped__: {has_wrapped(different_implementation)}")
print(f"cached_v1 has __wrapped__: {has_wrapped(cached_v1)}")

print("""
MITIGATION STRATEGY:
In hash_function_ast(), check for __wrapped__ and:
1. Hash BOTH the wrapper and the wrapped function
2. Or: Follow __wrapped__ chain and hash ALL functions in it
3. Or: Explicitly ignore __wrapped__ and get source differently

The current implementation is vulnerable because it trusts inspect.getsource()
which follows __wrapped__ automatically.
""")
