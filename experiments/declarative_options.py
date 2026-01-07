"""Explore declarative syntax options for matrix stages."""

from __future__ import annotations
import functools
from typing import Any, Callable, TypeVar, Sequence
from dataclasses import dataclass
from pydantic import BaseModel

F = TypeVar("F", bound=Callable[..., Any])

# Mock registry for testing
REGISTRATIONS: list[dict] = []

def mock_register(func, name, deps, outs, **kwargs):
    REGISTRATIONS.append({'func': func, 'name': name, 'deps': deps, 'outs': outs, **kwargs})


# =============================================================================
# OPTION A: `variants` dict on @stage
# =============================================================================
print("=" * 70)
print("OPTION A: variants dict on @stage")
print("=" * 70)

@dataclass
class stage_option_a:
    deps: Sequence[str] = ()
    outs: Sequence[str] = ()
    variants: dict[str, dict[str, Any]] | None = None

    def __call__(self, func: F) -> F:
        if self.variants is None:
            mock_register(func, func.__name__, list(self.deps), list(self.outs))
        else:
            for variant_name, spec in self.variants.items():
                bound = functools.partial(func, variant_name)
                mock_register(
                    bound,
                    f"{func.__name__}@{variant_name}",
                    spec.get('deps', list(self.deps)),
                    spec.get('outs', list(self.outs)),
                )
        return func

REGISTRATIONS.clear()

@stage_option_a(
    variants={
        'current': {'deps': ['data/current.csv'], 'outs': ['out/current.json']},
        'legacy': {'deps': ['data/legacy.csv', 'extra.csv'], 'outs': ['out/legacy.json']},
    }
)
def process_a(variant: str):
    """Single global function, variant passed as first arg."""
    return f"Processing {variant}"

print("Syntax:")
print('''
@stage(
    variants={
        'current': {'deps': ['data/current.csv'], 'outs': ['out/current.json']},
        'legacy': {'deps': ['data/legacy.csv', 'extra.csv'], 'outs': ['out/legacy.json']},
    }
)
def process(variant: str):
    ...
''')
print(f"Registered: {[r['name'] for r in REGISTRATIONS]}")
for r in REGISTRATIONS:
    result = r['func']()
    print(f"  {r['name']}: deps={r['deps']}, call()={result}")


# =============================================================================
# OPTION B: @stage.foreach() classmethod
# =============================================================================
print("\n" + "=" * 70)
print("OPTION B: @stage.foreach() classmethod")
print("=" * 70)

@dataclass
class stage_option_b:
    deps: Sequence[str] = ()
    outs: Sequence[str] = ()

    def __call__(self, func: F) -> F:
        mock_register(func, func.__name__, list(self.deps), list(self.outs))
        return func

    @classmethod
    def foreach(cls, specs: list[dict[str, Any]]) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            for spec in specs:
                name = spec['name']
                bound = functools.partial(func, name)
                mock_register(
                    bound,
                    f"{func.__name__}@{name}",
                    spec.get('deps', []),
                    spec.get('outs', []),
                )
            return func
        return decorator

REGISTRATIONS.clear()

@stage_option_b.foreach([
    {'name': 'current', 'deps': ['data/current.csv'], 'outs': ['out/current.json']},
    {'name': 'legacy', 'deps': ['data/legacy.csv', 'extra.csv'], 'outs': ['out/legacy.json']},
])
def process_b(variant: str):
    return f"Processing {variant}"

print("Syntax:")
print('''
@stage.foreach([
    {'name': 'current', 'deps': ['data/current.csv'], 'outs': ['out/current.json']},
    {'name': 'legacy', 'deps': ['data/legacy.csv', 'extra.csv'], 'outs': ['out/legacy.json']},
])
def process(variant: str):
    ...
''')
print(f"Registered: {[r['name'] for r in REGISTRATIONS]}")
for r in REGISTRATIONS:
    result = r['func']()
    print(f"  {r['name']}: deps={r['deps']}, call()={result}")


# =============================================================================
# OPTION C: Pydantic model for variant config
# =============================================================================
print("\n" + "=" * 70)
print("OPTION C: Pydantic model variants (full config object)")
print("=" * 70)

class VariantConfig(BaseModel):
    """User defines their own variant config model."""
    name: str
    deps: list[str]
    outs: list[str]
    # Can add any custom fields!
    extra_param: str = "default"

@dataclass
class stage_option_c:
    variants: list[BaseModel] | None = None

    def __call__(self, func: F) -> F:
        if self.variants is None:
            mock_register(func, func.__name__, [], [])
        else:
            for variant in self.variants:
                # Pass the whole config object to the function
                bound = functools.partial(func, variant)
                mock_register(
                    bound,
                    f"{func.__name__}@{variant.name}",
                    variant.deps,
                    variant.outs,
                )
        return func

REGISTRATIONS.clear()

@stage_option_c(variants=[
    VariantConfig(name='current', deps=['data/current.csv'], outs=['out/current.json']),
    VariantConfig(name='legacy', deps=['data/legacy.csv', 'extra.csv'], outs=['out/legacy.json'], extra_param='special'),
])
def process_c(config: VariantConfig):
    return f"Processing {config.name} with extra={config.extra_param}"

print("Syntax:")
print('''
class VariantConfig(BaseModel):
    name: str
    deps: list[str]
    outs: list[str]
    extra_param: str = "default"

@stage(variants=[
    VariantConfig(name='current', deps=['data/current.csv'], outs=['out/current.json']),
    VariantConfig(name='legacy', deps=['data/legacy.csv', 'extra.csv'], outs=['out/legacy.json'], extra_param='special'),
])
def process(config: VariantConfig):
    ...
''')
print(f"Registered: {[r['name'] for r in REGISTRATIONS]}")
for r in REGISTRATIONS:
    result = r['func']()
    print(f"  {r['name']}: deps={r['deps']}, call()={result}")


# =============================================================================
# OPTION D: Generator/callback expansion
# =============================================================================
print("\n" + "=" * 70)
print("OPTION D: Generator-based expansion")
print("=" * 70)

@dataclass
class stage_option_d:
    expand: Callable[[], list[dict]] | None = None

    def __call__(self, func: F) -> F:
        if self.expand is None:
            mock_register(func, func.__name__, [], [])
        else:
            for spec in self.expand():
                name = spec['name']
                bound = functools.partial(func, name)
                mock_register(
                    bound,
                    f"{func.__name__}@{name}",
                    spec.get('deps', []),
                    spec.get('outs', []),
                )
        return func

REGISTRATIONS.clear()

def process_variants():
    """Generator that yields variant specs."""
    base_deps = ['common.csv']
    for env in ['dev', 'prod']:
        extra = ['secrets.csv'] if env == 'prod' else []
        yield {
            'name': env,
            'deps': base_deps + extra,
            'outs': [f'output_{env}.json'],
        }

@stage_option_d(expand=process_variants)
def process_d(variant: str):
    return f"Processing {variant}"

print("Syntax:")
print('''
def process_variants():
    base_deps = ['common.csv']
    for env in ['dev', 'prod']:
        extra = ['secrets.csv'] if env == 'prod' else []
        yield {'name': env, 'deps': base_deps + extra, 'outs': [f'output_{env}.json']}

@stage(expand=process_variants)
def process(variant: str):
    ...
''')
print(f"Registered: {[r['name'] for r in REGISTRATIONS]}")
for r in REGISTRATIONS:
    result = r['func']()
    print(f"  {r['name']}: deps={r['deps']}, call()={result}")


# =============================================================================
# OPTION E: Hybrid - simple matrix + per-variant overrides
# =============================================================================
print("\n" + "=" * 70)
print("OPTION E: Simple matrix with overrides")
print("=" * 70)

@dataclass
class stage_option_e:
    deps: Sequence[str] = ()
    outs: Sequence[str] = ()
    matrix: dict[str, list[Any]] | None = None
    overrides: dict[str, dict[str, Any]] | None = None

    def __call__(self, func: F) -> F:
        if self.matrix is None:
            mock_register(func, func.__name__, list(self.deps), list(self.outs))
        else:
            # Simple single-dimension matrix for now
            param_name = list(self.matrix.keys())[0]
            values = self.matrix[param_name]
            overrides = self.overrides or {}

            for val in values:
                bound = functools.partial(func, val)
                # Apply overrides if present
                variant_deps = overrides.get(val, {}).get('deps', list(self.deps))
                variant_outs = overrides.get(val, {}).get('outs', list(self.outs))
                # Interpolate {param} in deps/outs
                variant_deps = [d.format(**{param_name: val}) for d in variant_deps]
                variant_outs = [o.format(**{param_name: val}) for o in variant_outs]

                mock_register(
                    bound,
                    f"{func.__name__}@{val}",
                    variant_deps,
                    variant_outs,
                )
        return func

REGISTRATIONS.clear()

@stage_option_e(
    matrix={'env': ['dev', 'staging', 'prod']},
    deps=['data/{env}.csv'],
    outs=['output/{env}.json'],
    overrides={
        'prod': {'deps': ['data/prod.csv', 'secrets.csv']},  # Extra dep for prod!
    }
)
def process_e(env: str):
    return f"Processing {env}"

print("Syntax:")
print('''
@stage(
    matrix={'env': ['dev', 'staging', 'prod']},
    deps=['data/{env}.csv'],
    outs=['output/{env}.json'],
    overrides={
        'prod': {'deps': ['data/prod.csv', 'secrets.csv']},  # Extra dep!
    }
)
def process(env: str):
    ...
''')
print(f"Registered: {[r['name'] for r in REGISTRATIONS]}")
for r in REGISTRATIONS:
    result = r['func']()
    print(f"  {r['name']}: deps={r['deps']}, call()={result}")


# =============================================================================
# OPTION F: Named tuple / dataclass variants (most Pythonic?)
# =============================================================================
print("\n" + "=" * 70)
print("OPTION F: Dataclass variants with unpacking")
print("=" * 70)

@dataclass
class Variant:
    """Lightweight variant spec."""
    name: str
    deps: list[str]
    outs: list[str]

@dataclass
class stage_option_f:
    variants: list[Variant] | None = None

    def __call__(self, func: F) -> F:
        if self.variants is None:
            mock_register(func, func.__name__, [], [])
        else:
            for v in self.variants:
                bound = functools.partial(func, v.name)
                mock_register(bound, f"{func.__name__}@{v.name}", v.deps, v.outs)
        return func

REGISTRATIONS.clear()

@stage_option_f(variants=[
    Variant('current', deps=['data/current.csv'], outs=['out/current.json']),
    Variant('legacy', deps=['data/legacy.csv', 'extra.csv'], outs=['out/legacy.json']),
])
def process_f(variant: str):
    return f"Processing {variant}"

print("Syntax:")
print('''
@stage(variants=[
    Variant('current', deps=['data/current.csv'], outs=['out/current.json']),
    Variant('legacy', deps=['data/legacy.csv', 'extra.csv'], outs=['out/legacy.json']),
])
def process(variant: str):
    ...
''')
print(f"Registered: {[r['name'] for r in REGISTRATIONS]}")
for r in REGISTRATIONS:
    result = r['func']()
    print(f"  {r['name']}: deps={r['deps']}, call()={result}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Declarative Options Comparison")
print("=" * 70)
print("""
| Option | Syntax Style              | Per-variant deps | Verbosity | Type Safety |
|--------|---------------------------|------------------|-----------|-------------|
| A      | variants={...} dict       | ✅ Full          | Medium    | ❌ dict     |
| B      | @stage.foreach([...])     | ✅ Full          | Medium    | ❌ dict     |
| C      | variants=[Model(...)]     | ✅ Full          | High      | ✅ Pydantic |
| D      | expand=generator_func     | ✅ Full          | Medium    | ❌ dict     |
| E      | matrix + overrides        | ✅ Via override  | Low       | ❌ dict     |
| F      | variants=[Variant(...)]   | ✅ Full          | Medium    | ✅ dataclass|

Key insights:
- Options A, B, F are very similar - just different syntax for list of specs
- Option C (Pydantic) allows custom variant fields beyond deps/outs
- Option D (generator) is most flexible for dynamic variant generation
- Option E (matrix + overrides) is best for mostly-uniform variants
""")
