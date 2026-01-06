"""More advanced declarative options - targeting the user's exact use case."""

from __future__ import annotations
import functools
from typing import Any, Callable, TypeVar, Sequence
from dataclasses import dataclass
from pydantic import BaseModel

F = TypeVar("F", bound=Callable[..., Any])
REGISTRATIONS: list[dict] = []

def mock_register(func, name, deps, outs, params=None, **kwargs):
    REGISTRATIONS.append({'func': func, 'name': name, 'deps': deps, 'outs': outs, 'params': params})


# User's target use case:
# - 2 dimensions: var_one (affects deps), var_two
# - deps vary by var_one (some have extra files)
# - outs depend on both variables
# - params is a pre-configured instance

class Params(BaseModel):
    var_three: str = "value3"


# =============================================================================
# OPTION G: Inline dict comprehension (most Pythonic?)
# =============================================================================
print("=" * 70)
print("OPTION G: Inline dict comprehension")
print("=" * 70)

base_deps = ["foo.csv", "bar.csv"]
variant_deps = {
    "value1": base_deps,
    "value2": [*base_deps, "value5.csv", "value6.csv"],
}

@dataclass
class stage_option_g:
    variants: dict[str, dict[str, Any]] | None = None

    def __call__(self, func: F) -> F:
        if self.variants:
            for name, spec in self.variants.items():
                bound = functools.partial(func, name)
                mock_register(bound, f"{func.__name__}@{name}", **spec)
        return func

REGISTRATIONS.clear()

@stage_option_g(variants={
    f"{v1}_{v2}": {
        'deps': variant_deps[v1],
        'outs': [f"{v1}_processed.csv", f"{v2}_processed.csv"],
        'params': Params(var_three=v1 + v2),
    }
    for v1 in ["value1", "value2"]
    for v2 in ["value3", "value4"]
})
def process_g(variant: str):
    return f"Processing {variant}"

print("Syntax:")
print('''
variant_deps = {"value1": base_deps, "value2": [*base_deps, "extra.csv"]}

@stage(variants={
    f"{v1}_{v2}": {
        'deps': variant_deps[v1],
        'outs': [f"{v1}_out.csv", f"{v2}_out.csv"],
        'params': Params(var_three=v1 + v2),
    }
    for v1 in ["value1", "value2"]
    for v2 in ["value3", "value4"]
})
def process(variant: str):
    ...
''')
print(f"Registered {len(REGISTRATIONS)} stages:")
for r in REGISTRATIONS:
    print(f"  {r['name']}: deps={r['deps']}, params.var_three={r['params'].var_three}")


# =============================================================================
# OPTION H: Helper function to generate variants
# =============================================================================
print("\n" + "=" * 70)
print("OPTION H: Helper function + variants parameter")
print("=" * 70)

def make_variants(
    var_one_options: list[tuple[str, list[str]]],  # (name, deps) pairs
    var_two_options: list[str],
) -> dict[str, dict]:
    """Generate variant specs from two dimensions."""
    return {
        f"{v1}_{v2}": {
            'deps': deps,
            'outs': [f"{v1}_processed.csv", f"{v2}_processed.csv"],
            'params': Params(var_three=v1 + v2),
        }
        for v1, deps in var_one_options
        for v2 in var_two_options
    }

REGISTRATIONS.clear()

@stage_option_g(variants=make_variants(
    var_one_options=[
        ("value1", ["foo.csv", "bar.csv"]),
        ("value2", ["foo.csv", "bar.csv", "value5.csv", "value6.csv"]),
    ],
    var_two_options=["value3", "value4"],
))
def process_h(variant: str):
    return f"Processing {variant}"

print("Syntax:")
print('''
def make_variants(var_one_options, var_two_options) -> dict[str, dict]:
    return {
        f"{v1}_{v2}": {'deps': deps, 'outs': [...], 'params': Params(...)}
        for v1, deps in var_one_options
        for v2 in var_two_options
    }

@stage(variants=make_variants(
    var_one_options=[("value1", base_deps), ("value2", extended_deps)],
    var_two_options=["value3", "value4"],
))
def process(variant: str):
    ...
''')
print(f"Registered {len(REGISTRATIONS)} stages:")
for r in REGISTRATIONS:
    print(f"  {r['name']}: deps={r['deps']}")


# =============================================================================
# OPTION I: Tuple unpacking in function signature
# =============================================================================
print("\n" + "=" * 70)
print("OPTION I: Multiple variant params in function signature")
print("=" * 70)

@dataclass
class stage_option_i:
    variants: dict[str, dict[str, Any]] | None = None

    def __call__(self, func: F) -> F:
        if self.variants:
            for name, spec in self.variants.items():
                # Parse name to get individual params: "value1_value3" -> ["value1", "value3"]
                params_from_name = name.split("_")
                bound = functools.partial(func, *params_from_name)
                mock_register(bound, f"{func.__name__}@{name}", **spec)
        return func

REGISTRATIONS.clear()

@stage_option_i(variants={
    f"{v1}_{v2}": {
        'deps': variant_deps[v1],
        'outs': [f"{v1}_processed.csv", f"{v2}_processed.csv"],
        'params': Params(var_three=v1 + v2),
    }
    for v1 in ["value1", "value2"]
    for v2 in ["value3", "value4"]
})
def process_i(var_one: str, var_two: str):
    """Function receives BOTH variant dimensions as separate args."""
    return f"Processing var_one={var_one}, var_two={var_two}"

print("Syntax:")
print('''
@stage(variants={
    f"{v1}_{v2}": {...}
    for v1 in ["value1", "value2"]
    for v2 in ["value3", "value4"]
})
def process(var_one: str, var_two: str):  # Gets both dimensions!
    ...
''')
print(f"Registered {len(REGISTRATIONS)} stages:")
for r in REGISTRATIONS:
    result = r['func']()
    print(f"  {r['name']}: {result}")


# =============================================================================
# OPTION J: Named variant specs with explicit params
# =============================================================================
print("\n" + "=" * 70)
print("OPTION J: VariantSpec dataclass with param binding")
print("=" * 70)

@dataclass
class VariantSpec:
    """Explicit variant specification."""
    name: str
    deps: list[str]
    outs: list[str]
    params: BaseModel | None = None
    bind: dict[str, Any] | None = None  # Values to pass to function

@dataclass
class stage_option_j:
    variants: list[VariantSpec] | None = None

    def __call__(self, func: F) -> F:
        if self.variants:
            for v in self.variants:
                if v.bind:
                    bound = functools.partial(func, **v.bind)
                else:
                    bound = functools.partial(func, v.name)
                mock_register(bound, f"{func.__name__}@{v.name}", v.deps, v.outs, v.params)
        return func

REGISTRATIONS.clear()

@stage_option_j(variants=[
    VariantSpec(
        name="value1_value3",
        deps=["foo.csv", "bar.csv"],
        outs=["value1_processed.csv", "value3_processed.csv"],
        params=Params(var_three="value1value3"),
        bind={"var_one": "value1", "var_two": "value3"},
    ),
    VariantSpec(
        name="value2_value4",
        deps=["foo.csv", "bar.csv", "value5.csv", "value6.csv"],
        outs=["value2_processed.csv", "value4_processed.csv"],
        params=Params(var_three="value2value4"),
        bind={"var_one": "value2", "var_two": "value4"},
    ),
])
def process_j(var_one: str, var_two: str):
    return f"Processing {var_one}, {var_two}"

print("Syntax:")
print('''
@stage(variants=[
    VariantSpec(
        name="value1_value3",
        deps=["foo.csv", "bar.csv"],
        outs=["value1_out.csv", "value3_out.csv"],
        params=Params(var_three="value1value3"),
        bind={"var_one": "value1", "var_two": "value3"},
    ),
    ...
])
def process(var_one: str, var_two: str):
    ...
''')
print(f"Registered {len(REGISTRATIONS)} stages:")
for r in REGISTRATIONS:
    result = r['func']()
    print(f"  {r['name']}: {result}")


# =============================================================================
# OPTION K: Simplest possible - just list of complete specs
# =============================================================================
print("\n" + "=" * 70)
print("OPTION K: Simplest - list of complete dicts")
print("=" * 70)

@dataclass
class stage_option_k:
    variants: list[dict[str, Any]] | None = None

    def __call__(self, func: F) -> F:
        if self.variants:
            for v in self.variants:
                name = v['name']
                bind = v.get('bind', {})
                if bind:
                    bound = functools.partial(func, **bind)
                else:
                    bound = functools.partial(func, name)
                mock_register(
                    bound,
                    f"{func.__name__}@{name}",
                    v.get('deps', []),
                    v.get('outs', []),
                    v.get('params'),
                )
        return func

REGISTRATIONS.clear()

# Can be generated programmatically
variants_list = [
    {
        'name': f"{v1}_{v2}",
        'deps': variant_deps[v1],
        'outs': [f"{v1}_processed.csv", f"{v2}_processed.csv"],
        'params': Params(var_three=v1 + v2),
        'bind': {'var_one': v1, 'var_two': v2},
    }
    for v1 in ["value1", "value2"]
    for v2 in ["value3", "value4"]
]

@stage_option_k(variants=variants_list)
def process_k(var_one: str, var_two: str):
    return f"Processing {var_one}, {var_two}"

print("Syntax:")
print('''
variants = [
    {
        'name': f"{v1}_{v2}",
        'deps': variant_deps[v1],
        'outs': [f"{v1}_out.csv", f"{v2}_out.csv"],
        'params': Params(var_three=v1 + v2),
        'bind': {'var_one': v1, 'var_two': v2},
    }
    for v1 in ["value1", "value2"]
    for v2 in ["value3", "value4"]
]

@stage(variants=variants)
def process(var_one: str, var_two: str):
    ...
''')
print(f"Registered {len(REGISTRATIONS)} stages:")
for r in REGISTRATIONS:
    result = r['func']()
    print(f"  {r['name']}: deps={r['deps'][:2]}..., params.var_three={r['params'].var_three}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: Which is cleanest for the 2-dimension use case?")
print("=" * 70)
print("""
USER'S ORIGINAL (loop-based):
  for v1, deps in [...]:
      for v2 in [...]:
          @stage(name=f"...", deps=deps, outs=[...], params=Params(...))
          def process(v1=v1, v2=v2): ...

OPTION G (inline dict comprehension):
  @stage(variants={f"{v1}_{v2}": {...} for v1 in [...] for v2 in [...]})
  def process(variant: str): ...

OPTION H (helper function):
  @stage(variants=make_variants(var1_opts, var2_opts))
  def process(variant: str): ...

OPTION I (tuple unpacking):
  @stage(variants={...})
  def process(var_one: str, var_two: str): ...  # Gets both!

OPTION K (list + bind):
  @stage(variants=[{'name': ..., 'bind': {'v1': ..., 'v2': ...}}])
  def process(var_one: str, var_two: str): ...

RECOMMENDATION:
- For simple cases: OPTION G (dict comprehension) is cleanest
- For complex cases: OPTION K (list with bind) is most explicit
- Key insight: 'bind' parameter lets you pass multiple values to function
""")
