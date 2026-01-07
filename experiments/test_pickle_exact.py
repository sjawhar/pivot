"""Test picklability with user's exact pattern."""

import pickle
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor


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


# User's exact pattern
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


print("Registered stages:", [r['name'] for r in REGISTRATIONS])
print()

# Test pickle each function
print("Pickle test for each registered function:")
for i, reg in enumerate(REGISTRATIONS):
    try:
        pickled = pickle.dumps(reg['func'])
        unpickled = pickle.loads(pickled)
        print(f"  [{i}] {reg['name']}: ✅ pickle OK")
    except Exception as e:
        print(f"  [{i}] {reg['name']}: ❌ pickle FAILED - {e}")

print()

# Test with ProcessPoolExecutor
print("ProcessPoolExecutor test:")


def run_stage(func, deps, outs):
    """Worker function to run a stage."""
    return func(deps, outs)


# Only test the last registration (the one that pickles)
last_reg = REGISTRATIONS[-1]
first_reg = REGISTRATIONS[0]

try:
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_stage, last_reg['func'], last_reg['deps'], last_reg['outs'])
        result = future.result(timeout=5)
        print(f"  Last stage ({last_reg['name']}): ✅ executed in worker")
except Exception as e:
    print(f"  Last stage ({last_reg['name']}): ❌ failed - {e}")

try:
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_stage, first_reg['func'], first_reg['deps'], first_reg['outs'])
        result = future.result(timeout=5)
        print(f"  First stage ({first_reg['name']}): ✅ executed in worker")
except Exception as e:
    print(f"  First stage ({first_reg['name']}): ❌ failed - {e}")
