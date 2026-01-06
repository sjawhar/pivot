"""Test global function + loop registration approach."""

import pickle
from pydantic import BaseModel
from loky import get_reusable_executor
from concurrent.futures import ProcessPoolExecutor
import functools


class Params(BaseModel):
    var_three: str = "value3"


REGISTRATIONS: list[dict] = []


def stage(name: str, deps: list[str], outs: list[str], params: BaseModel | None = None):
    def decorator(func):
        REGISTRATIONS.append({
            'name': name,
            'deps': deps,
            'outs': outs,
            'params': params,
            'func': func,
        })
        return func
    return decorator


# APPROACH 1: Global function, registered multiple times in loop
# The function is defined ONCE at module level
def process_global(variant: str, deps: list[str], outs: list[str]):
    """Global function that takes variant as first arg."""
    return f"Processing variant={variant}, deps={deps}"


print("=" * 60)
print("APPROACH 1: Global function + loop registration (with partial)")
print("=" * 60)

REGISTRATIONS.clear()
base_deps = ["foo.csv", "bar.csv"]

for var_one, var_deps in [
    ("value1", base_deps),
    ("value2", [*base_deps, "value5.csv", "value6.csv"]),
]:
    for var_two in ["value3", "value4"]:
        # Create a partial that binds the variant
        bound_func = functools.partial(process_global, f"{var_one}_{var_two}")

        # Register the partial
        stage(
            name=f"process_{var_one}_{var_two}",
            deps=var_deps,
            outs=[f"{var_one}_processed.csv", f"{var_two}_processed.csv"],
            params=Params(var_three=var_one + var_two),
        )(bound_func)

print(f"Registered {len(REGISTRATIONS)} stages")

# Test standard pickle
print("\nStandard pickle test:")
for reg in REGISTRATIONS:
    try:
        pickled = pickle.dumps(reg['func'])
        unpickled = pickle.loads(pickled)
        result = unpickled(deps=['test'], outs=['out'])
        print(f"  {reg['name']}: ✅ {result}")
    except Exception as e:
        print(f"  {reg['name']}: ❌ {e}")

# Test ProcessPoolExecutor
print("\nProcessPoolExecutor test:")
with ProcessPoolExecutor(max_workers=1) as executor:
    for reg in REGISTRATIONS[:2]:  # Just test first 2
        try:
            future = executor.submit(reg['func'], deps=reg['deps'], outs=reg['outs'])
            result = future.result(timeout=5)
            print(f"  {reg['name']}: ✅ {result}")
        except Exception as e:
            print(f"  {reg['name']}: ❌ {e}")


# APPROACH 2: Global function, use @stage decorator directly with name override
print("\n" + "=" * 60)
print("APPROACH 2: Global function + @stage with name (current proposal)")
print("=" * 60)

REGISTRATIONS.clear()


# Would need a way to pass variant to the function
# Option A: Params object with variant info
def process_with_params(params: Params):
    """Function that gets variant from params."""
    return f"Processing with params.var_three={params.var_three}"


for var_one, var_deps in [
    ("value1", base_deps),
    ("value2", [*base_deps, "value5.csv", "value6.csv"]),
]:
    for var_two in ["value3", "value4"]:
        # The stage function doesn't need to capture the variant
        # because it gets it from params
        stage(
            name=f"process_{var_one}_{var_two}",
            deps=var_deps,
            outs=[f"{var_one}_processed.csv", f"{var_two}_processed.csv"],
            params=Params(var_three=var_one + var_two),
        )(process_with_params)

print(f"Registered {len(REGISTRATIONS)} stages")

# Test pickle
print("\nStandard pickle test:")
for reg in REGISTRATIONS:
    try:
        pickled = pickle.dumps(reg['func'])
        unpickled = pickle.loads(pickled)
        result = unpickled(params=reg['params'])
        print(f"  {reg['name']}: ✅ {result}")
    except Exception as e:
        print(f"  {reg['name']}: ❌ {e}")


# APPROACH 3: User's exact syntax with loky (should work)
print("\n" + "=" * 60)
print("APPROACH 3: User's exact syntax (relies on loky/cloudpickle)")
print("=" * 60)

REGISTRATIONS.clear()

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
            return f"processed deps={deps}"

print(f"Registered {len(REGISTRATIONS)} stages")

# Test with loky
print("\nLoky test:")
executor = get_reusable_executor(max_workers=2)
for reg in REGISTRATIONS:
    try:
        future = executor.submit(reg['func'], deps=reg['deps'], outs=reg['outs'])
        result = future.result(timeout=5)
        print(f"  {reg['name']}: ✅ {result}")
    except Exception as e:
        print(f"  {reg['name']}: ❌ {e}")
executor.shutdown(wait=True)


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Approach 1 (Global + partial): ✅ Works with standard pickle
Approach 2 (Global + params):  ✅ Works with standard pickle, variant in params
Approach 3 (Loop-defined):     ✅ Works with loky/cloudpickle ONLY

Since Pivot uses loky, all approaches work!

However, Approach 1/2 are safer because:
- They work with ANY executor (standard pickle compatible)
- The function is defined once at module level (clearer code)
- Variant is explicitly bound via partial or params
""")
