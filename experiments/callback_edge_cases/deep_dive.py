# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportUnknownLambdaType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
Deep dive into the vulnerabilities found.
"""

import functools
import sys
import inspect

sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint


print("="*70)
print("VULNERABILITY 1: functools.partial")
print("="*70)
print("""
WHY: functools.partial creates a new callable that wraps the original.
The partial object is passed as a nonlocal to the inner function.
However, when fingerprinting processes the partial:
1. It's callable - check
2. is_user_code(partial_v1) - what does this return?
""")

def base_function(multiplier, x):
    return x * multiplier

partial_v1 = functools.partial(base_function, 2)
partial_v2 = functools.partial(base_function, 3)

print(f"is_user_code(partial_v1): {fingerprint.is_user_code(partial_v1)}")
print(f"type(partial_v1): {type(partial_v1)}")
print(f"partial_v1.__module__: {getattr(partial_v1, '__module__', 'N/A')}")

# What does getclosurevars see?
def make_stage(p):
    def stage():
        return p(10)
    return stage

stage1 = make_stage(partial_v1)
cv = inspect.getclosurevars(stage1)
print(f"\nclosurevars of stage using partial:")
print(f"  globals: {cv.globals}")
print(f"  nonlocals: {cv.nonlocals}")
print(f"  builtins: {list(cv.builtins.keys())[:5]}...")

# The partial is in nonlocals as 'p', but is_user_code returns False!
# Because functools.partial is from stdlib

print("""
CONCLUSION: functools.partial objects are NOT user code (stdlib),
so they're skipped entirely. The wrapped function and arguments are lost.

IMPACT: HIGH - Any stage using functools.partial will not track:
- Changes to the wrapped function
- Changes to the partial arguments
""")


print("\n" + "="*70)
print("VULNERABILITY 2: Multi-layer callbacks")
print("="*70)
print("""
WHY: When a callback is passed through another function that returns
a wrapper, the inner callback becomes a nonlocal of the wrapper.
""")

def _inner_callback():
    return 42

def middle_layer(callback):
    def wrapper():
        return callback() + 1
    return wrapper

wrapped = middle_layer(_inner_callback)

print(f"type(wrapped): {type(wrapped)}")
print(f"is_user_code(wrapped): {fingerprint.is_user_code(wrapped)}")

cv = inspect.getclosurevars(wrapped)
print(f"\nclosurevars of wrapped:")
print(f"  globals: {cv.globals}")
print(f"  nonlocals: {cv.nonlocals}")

# The 'callback' nonlocal IS the _inner_callback function
# Let's check if fingerprinting recurses into it
fp = fingerprint.get_stage_fingerprint(wrapped)
print(f"\nFingerprint of wrapped: {fp}")

# Now change the inner callback and see if it's detected
def _inner_callback_v2():
    return 100

wrapped_v2 = middle_layer(_inner_callback_v2)
fp2 = fingerprint.get_stage_fingerprint(wrapped_v2)
print(f"Fingerprint of wrapped_v2: {fp2}")

if fp == fp2:
    print("VULNERABILITY CONFIRMED: Different inner callbacks, same fingerprint!")
else:
    print("Actually detected - need to investigate stage usage pattern")

print("""
CONCLUSION: The wrapped function DOES capture and track its callback nonlocal.
The issue is when a STAGE uses wrapped_callback - does the stage's fingerprint
include the inner callback?
""")

def stage_multi_layer():
    return wrapped()

fp_stage = fingerprint.get_stage_fingerprint(stage_multi_layer)
print(f"\nStage fingerprint: {fp_stage}")
print(f"Has callback key: {'callback' in str(fp_stage)}")

print("""
The stage fingerprint DOES include 'func:callback' (the inner callback).
The "vulnerability" was a false positive - checking if wrapped_callback is
captured IS sufficient because fingerprinting recurses into its closure.
""")


print("\n" + "="*70)
print("VULNERABILITY 3: Class attribute callbacks")
print("="*70)

class Config:
    callback = None

def _cb_v1():
    return 1

def _cb_v2():
    return 2

Config.callback = _cb_v1

def stage_class_attr():
    return Config.callback()

# Check what fingerprinting sees
cv = inspect.getclosurevars(stage_class_attr)
print(f"closurevars of stage_class_attr:")
print(f"  globals: {cv.globals}")
print(f"  nonlocals: {cv.nonlocals}")

# Config is in globals. Let's see what fingerprinting does with it
fp1 = fingerprint.get_stage_fingerprint(stage_class_attr)
print(f"\nFingerprint with _cb_v1: {fp1}")

Config.callback = _cb_v2
fp2 = fingerprint.get_stage_fingerprint(stage_class_attr)
print(f"Fingerprint with _cb_v2: {fp2}")

print(f"\nFP1 == FP2: {fp1 == fp2}")

print("""
ANALYSIS:
- Config class is detected as a user-defined class instance? No, it's a type!
- The fingerprint shows 'class:Config' - it hashes the CLASS definition
- But Config.callback is a RUNTIME ATTRIBUTE, not part of the class definition
- The class body at parse time has `callback = None`, so that's what gets hashed

CONCLUSION: Class attributes that are modified at runtime are NOT tracked.
Only the original class definition is hashed.

IMPACT: HIGH - Any stage that relies on class attributes set at runtime
will not detect changes to those attributes.
""")


print("\n" + "="*70)
print("VULNERABILITY 4: functools.wraps")
print("="*70)

def original():
    return 1

@functools.wraps(original)
def wrapped_different():
    return 2  # Different code!

print(f"original.__name__: {original.__name__}")
print(f"wrapped_different.__name__: {wrapped_different.__name__}")
print(f"wrapped_different.__wrapped__: {getattr(wrapped_different, '__wrapped__', 'N/A')}")

fp_orig = fingerprint.get_stage_fingerprint(original)
fp_wrapped = fingerprint.get_stage_fingerprint(wrapped_different)

print(f"\noriginal fingerprint: {fp_orig}")
print(f"wrapped_different fingerprint: {fp_wrapped}")

# Let's check the hash_function_ast directly
h_orig = fingerprint.hash_function_ast(original)
h_wrapped = fingerprint.hash_function_ast(wrapped_different)
print(f"\nhash original: {h_orig}")
print(f"hash wrapped: {h_wrapped}")

# What source does getsource return?
print(f"\noriginal source:\n{inspect.getsource(original)}")
print(f"\nwrapped_different source:\n{inspect.getsource(wrapped_different)}")

print("""
WAIT - the sources ARE different! Let me check why the hashes are the same...
""")

# Actually, looking at the original test, both used `return 1` vs `return 2`
# But the test above shows DIFFERENT hashes!
# Let me recreate the exact scenario from the original test

print("\nRecreating original test scenario:")

def original_func():
    """Original function."""
    return 1

@functools.wraps(original_func)
def wrapped_func():
    """This is actually different."""
    return 2

fp1 = fingerprint.get_stage_fingerprint(original_func)
fp2 = fingerprint.get_stage_fingerprint(wrapped_func)
print(f"original_func FP: {fp1}")
print(f"wrapped_func FP: {fp2}")

# AHA! The key is 'self:original_func' in BOTH cases because
# functools.wraps copies __name__ from the original to the wrapped function!

print("""
ROOT CAUSE FOUND:
- functools.wraps copies __name__ from original to wrapped
- Both fingerprints use 'self:original_func' as the key
- The hash VALUES are different, but they're stored under the SAME KEY!
- So when comparing fingerprints as dicts, they appear equal if keys match

Actually wait, let's check the actual values...
""")

print(f"fp1['self:original_func']: {fp1['self:original_func']}")
print(f"fp2['self:original_func']: {fp2['self:original_func']}")
print(f"Values equal: {fp1['self:original_func'] == fp2['self:original_func']}")

# The values are actually equal! That means the AST hashes are equal.
# But the code is different... Let's check getsource

src_orig = inspect.getsource(original_func)
src_wrapped = inspect.getsource(wrapped_func)
print(f"\noriginal_func source:\n{src_orig}")
print(f"\nwrapped_func source:\n{src_wrapped}")

print("""
SMOKING GUN: inspect.getsource(wrapped_func) returns the SOURCE OF wrapped_func,
NOT original_func. But the AST hashes are equal...

Let me check what _normalize_ast does with docstrings...
""")

import ast

tree_orig = ast.parse(src_orig)
tree_wrapped = ast.parse(src_wrapped)

# The docstrings should be stripped, and function names normalized
# Let's see the dumps

print("Original AST dump (before normalize):")
print(ast.dump(tree_orig, indent=2)[:500])

print("\nWrapped AST dump (before normalize):")
print(ast.dump(tree_wrapped, indent=2)[:500])

print("""
AH HA! The issue is that _normalize_ast normalizes FUNCTION NAMES to 'func'.
So both functions become:
  def func():
      return X

But X is 1 vs 2, so they SHOULD be different...

Actually, looking at the test output from before, both returned 1!
Let me check the actual test functions used...
""")

# Look at what was ACTUALLY tested - in the original test file
# Both docstrings are different, but the CODE (return 1) is the same!
# @functools.wraps just changes metadata, but the test used the SAME return value

print("""
CORRECTION: In the original test:
- original_func() returns 1
- wrapped_func() returns 2 (different!)

But the hashes were reported equal. Let me verify one more time with fresh functions...
""")

def fresh_v1():
    return 111

@functools.wraps(fresh_v1)
def fresh_v2():
    return 222

h1 = fingerprint.hash_function_ast(fresh_v1)
h2 = fingerprint.hash_function_ast(fresh_v2)
print(f"fresh_v1 hash: {h1}")
print(f"fresh_v2 hash: {h2}")
print(f"Equal: {h1 == h2}")

print("""
They're DIFFERENT now! So the original test case must have been buggy.

Let me look at the EXACT code from the original test file...
""")


print("\n" + "="*70)
print("VULNERABILITY 5: Subscript callbacks (__getitem__)")
print("="*70)

class Container:
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

container = Container()
container["handler"] = _cb_v1

def stage_subscript():
    return container["handler"]()

cv = inspect.getclosurevars(stage_subscript)
print(f"closurevars of stage_subscript:")
print(f"  globals: {cv.globals}")

fp1 = fingerprint.get_stage_fingerprint(stage_subscript)
print(f"\nFingerprint with _cb_v1: {fp1}")

container["handler"] = _cb_v2
fp2 = fingerprint.get_stage_fingerprint(stage_subscript)
print(f"Fingerprint with _cb_v2: {fp2}")

print(f"\nFP1 == FP2: {fp1 == fp2}")

print("""
ANALYSIS:
- container is in globals as a user-defined class instance
- _process_instance_dependency hashes the CLASS definition (Container)
- It does NOT hash the INSTANCE STATE (container._data)
- So changing container["handler"] doesn't change the fingerprint

CONCLUSION: Instance state is not tracked, only class definitions.

IMPACT: HIGH - Any mutable container holding callbacks will not track changes.
""")


print("\n" + "="*70)
print("SUMMARY OF REAL VULNERABILITIES")
print("="*70)
print("""
1. functools.partial - NOT tracked (stdlib, not user code)
   - Severity: HIGH
   - Trigger: Using partial(my_func, arg) where arg changes
   - Impact: Stage won't re-run when partial arguments change

2. Multi-layer callbacks - ACTUALLY WORKS
   - The test was a false positive
   - Fingerprinting DOES recurse into closure variables

3. Class attribute callbacks - NOT tracked
   - Severity: HIGH
   - Trigger: Class.attr = different_callback at runtime
   - Impact: Stage won't re-run when class attribute changes

4. functools.wraps - NEED TO VERIFY
   - May have been a test bug
   - Needs more investigation

5. Subscript/Instance state callbacks - NOT tracked
   - Severity: HIGH
   - Trigger: container["key"] = different_callback
   - Impact: Stage won't re-run when instance state changes

6. getattr with string literals - NOT tracked (by design)
   - Severity: MEDIUM
   - Trigger: getattr(obj, "method_name")() where method_name is a string
   - Impact: Dynamic dispatch based on strings not tracked
""")
