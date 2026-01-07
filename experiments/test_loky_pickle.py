"""Test if loky/cloudpickle handles loop-defined functions."""

from pydantic import BaseModel
from loky import get_reusable_executor


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
            return f"processed {deps}"


print("Registered stages:", [r['name'] for r in REGISTRATIONS])
print()

# Test with loky (uses cloudpickle)
print("Loky/cloudpickle test:")


def run_stage(func, deps, outs):
    """Worker function to run a stage."""
    return func(deps, outs)


executor = get_reusable_executor(max_workers=2)

for i, reg in enumerate(REGISTRATIONS):
    try:
        future = executor.submit(run_stage, reg['func'], reg['deps'], reg['outs'])
        result = future.result(timeout=5)
        print(f"  [{i}] {reg['name']}: ✅ executed - {result}")
    except Exception as e:
        print(f"  [{i}] {reg['name']}: ❌ failed - {e}")

# Shutdown executor
executor.shutdown(wait=True)
