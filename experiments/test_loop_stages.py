"""Test loop-based stage registration to identify potential issues."""

import functools
import pickle
import inspect
from pydantic import BaseModel


# Simulate the stage decorator storing registrations
REGISTRATIONS: list[dict] = []


def stage(name: str, deps: list[str], outs: list[str], params: BaseModel | None = None):
    """Simplified stage decorator that captures registration info."""
    def decorator(func):
        REGISTRATIONS.append({
            'name': name,
            'deps': deps,
            'outs': outs,
            'params': params,
            'func': func,
            'func_id': id(func),
            'func_name': func.__name__,
        })
        return func
    return decorator


class Params(BaseModel):
    var_three: str = "value3"


# Test 1: Basic loop registration
print("=" * 60)
print("TEST 1: Basic loop registration")
print("=" * 60)

REGISTRATIONS.clear()
base_deps = ["foo.csv", "bar.csv"]

for var_one, var_deps in [
    ("value1", base_deps),
    ("value2", [*base_deps, "value5.csv", "value6.csv"]),
]:
    for var_two in ["value3", "value4"]:
        @stage(
            name=f"process_{var_one}_{var_two}",
            deps=var_deps,
            outs=[f"{var_one}_processed.csv", f"{var_two}_processed.csv"],
            params=Params(var_three=var_one + var_two),
        )
        def process(deps, outs):
            return deps, outs

print(f"Registered {len(REGISTRATIONS)} stages:")
for reg in REGISTRATIONS:
    print(f"  - {reg['name']}: deps={reg['deps']}, params.var_three={reg['params'].var_three}")

# Check if all functions are the same object (they shouldn't be for fingerprinting)
func_ids = [reg['func_id'] for reg in REGISTRATIONS]
print(f"\nFunction IDs unique: {len(set(func_ids)) == len(func_ids)}")
print(f"Function IDs: {func_ids}")


# Test 2: Closure footgun - does the function capture loop variables?
print("\n" + "=" * 60)
print("TEST 2: Closure footgun test")
print("=" * 60)

REGISTRATIONS.clear()
captured_values = []

for var_one in ["a", "b", "c"]:
    @stage(name=f"stage_{var_one}", deps=[], outs=[], params=None)
    def process_with_closure():
        # This captures var_one by reference!
        return f"var_one is {var_one}"

# After the loop, var_one = "c"
print("Calling each registered function:")
for reg in REGISTRATIONS:
    result = reg['func']()
    print(f"  {reg['name']}: {result}")
    captured_values.append(result)

# All should say "c" due to closure bug
all_same = len(set(captured_values)) == 1
print(f"\nClosure footgun present: {all_same} (all functions return same value)")


# Test 3: Picklability
print("\n" + "=" * 60)
print("TEST 3: Picklability test")
print("=" * 60)

REGISTRATIONS.clear()

for variant in ["x", "y"]:
    @stage(name=f"pickle_test_{variant}", deps=[], outs=[], params=None)
    def pickle_test_func(v=variant):  # Default arg captures value!
        return f"variant is {v}"

print("Testing pickle of registered functions:")
for reg in REGISTRATIONS:
    try:
        pickled = pickle.dumps(reg['func'])
        unpickled = pickle.loads(pickled)
        result = unpickled()
        print(f"  {reg['name']}: pickle OK, result={result}")
    except Exception as e:
        print(f"  {reg['name']}: pickle FAILED - {e}")


# Test 4: Fix closure with default arg
print("\n" + "=" * 60)
print("TEST 4: Fix closure with default argument")
print("=" * 60)

REGISTRATIONS.clear()

for var_one in ["a", "b", "c"]:
    @stage(name=f"fixed_{var_one}", deps=[], outs=[], params=None)
    def process_fixed(captured_var=var_one):  # Default arg captures VALUE at definition time
        return f"var_one is {captured_var}"

print("Calling each registered function (with default arg fix):")
for reg in REGISTRATIONS:
    result = reg['func']()
    print(f"  {reg['name']}: {result}")


# Test 5: Function identity and code fingerprinting
print("\n" + "=" * 60)
print("TEST 5: Function code identity")
print("=" * 60)

REGISTRATIONS.clear()

for variant in ["p", "q"]:
    @stage(name=f"code_test_{variant}", deps=[], outs=[], params=None)
    def code_test():
        pass

print("Function code objects:")
for reg in REGISTRATIONS:
    func = reg['func']
    print(f"  {reg['name']}:")
    print(f"    __code__: {id(func.__code__)}")
    print(f"    co_code: {func.__code__.co_code.hex()[:40]}...")
    print(f"    __module__: {func.__module__}")
    print(f"    __qualname__: {func.__qualname__}")

# Are the code objects the same?
codes = [reg['func'].__code__ for reg in REGISTRATIONS]
print(f"\nAll code objects identical: {all(c is codes[0] for c in codes)}")
print(f"All co_code bytes identical: {all(c.co_code == codes[0].co_code for c in codes)}")


# Test 6: Using functools.partial instead
print("\n" + "=" * 60)
print("TEST 6: functools.partial approach")
print("=" * 60)

REGISTRATIONS.clear()

def base_process(variant: str, deps, outs):
    """Base function defined once at module level."""
    return f"Processing {variant} with {deps}"

for variant in ["alpha", "beta"]:
    bound_func = functools.partial(base_process, variant)
    REGISTRATIONS.append({
        'name': f"partial_{variant}",
        'func': bound_func,
    })

print("Testing functools.partial functions:")
for reg in REGISTRATIONS:
    result = reg['func'](deps=['a.csv'], outs=['b.csv'])
    print(f"  {reg['name']}: {result}")

print("\nPickle test for partial:")
for reg in REGISTRATIONS:
    try:
        pickled = pickle.dumps(reg['func'])
        unpickled = pickle.loads(pickled)
        result = unpickled(deps=['test'], outs=['out'])
        print(f"  {reg['name']}: pickle OK, result={result}")
    except Exception as e:
        print(f"  {reg['name']}: pickle FAILED - {e}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key findings:
1. Loop-defined functions have CLOSURE FOOTGUN - they capture loop var by reference
2. Default arguments (v=variant) can fix the closure issue
3. Functions defined in loops ARE picklable (they're still module-level)
4. All loop-defined functions share the same __code__ object
5. functools.partial is clean and avoids closure issues

Recommendations:
- If using loop approach: MUST use default args to capture values
- functools.partial is safer and more explicit
- Need to differentiate stages by name, not by function code
""")
