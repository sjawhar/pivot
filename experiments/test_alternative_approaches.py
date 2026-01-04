#!/usr/bin/env python3
"""
Additional experiments exploring alternative approaches to function change detection.
"""
import ast
import hashlib
import inspect
import pickle
import textwrap
import types


def get_function_globals_hash(func):
    """
    Hash the actual values of globals referenced by the function.
    This catches global variable changes!
    """
    code = func.__code__
    
    global_names = set(code.co_names) - set(code.co_varnames)
    
    values = []
    for name in sorted(global_names):
        if name in func.__globals__:
            val = func.__globals__[name]
            if callable(val):
                values.append(f"{name}=<callable:{getattr(val, '__name__', 'unknown')}>")
            else:
                try:
                    values.append(f"{name}={repr(val)}")
                except Exception:
                    values.append(f"{name}=<unhashable>")
    
    return hashlib.sha256(str(values).encode()).hexdigest()[:16]


def get_function_closure_hash(func):
    """Hash closure values."""
    if func.__closure__ is None:
        return "no_closure"
    
    values = []
    for i, cell in enumerate(func.__closure__):
        try:
            values.append(repr(cell.cell_contents))
        except ValueError:
            values.append("<empty_cell>")
    
    return hashlib.sha256(str(values).encode()).hexdigest()[:16]


def get_defaults_hash(func):
    """Hash default argument values."""
    parts = []
    if func.__defaults__:
        for d in func.__defaults__:
            try:
                parts.append(repr(d))
            except Exception:
                parts.append("<unhashable>")
    if func.__kwdefaults__:
        for k, v in sorted(func.__kwdefaults__.items()):
            try:
                parts.append(f"{k}={repr(v)}")
            except Exception:
                parts.append(f"{k}=<unhashable>")
    
    if not parts:
        return "no_defaults"
    
    return hashlib.sha256(str(parts).encode()).hexdigest()[:16]


def comprehensive_function_hash(func):
    """
    A more comprehensive hash that includes:
    - AST of the function body
    - Values of referenced globals (non-callable)
    - Closure cell values
    - Default argument values
    """
    components = []
    
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        components.append(("ast", ast.dump(tree)))
    except Exception:
        components.append(("ast", "unavailable"))
    
    code = func.__code__
    global_names = set(code.co_names) - set(code.co_varnames)
    global_vals = []
    for name in sorted(global_names):
        if name in func.__globals__:
            val = func.__globals__[name]
            if not callable(val):
                try:
                    global_vals.append(f"{name}={repr(val)}")
                except Exception:
                    pass
    components.append(("globals", str(global_vals)))
    
    if func.__closure__:
        closure_vals = []
        for cell in func.__closure__:
            try:
                val = cell.cell_contents
                if not callable(val):
                    closure_vals.append(repr(val))
            except ValueError:
                pass
        components.append(("closure", str(closure_vals)))
    
    if func.__defaults__:
        try:
            components.append(("defaults", repr(func.__defaults__)))
        except Exception:
            pass
    if func.__kwdefaults__:
        try:
            components.append(("kwdefaults", repr(func.__kwdefaults__)))
        except Exception:
            pass
    
    combined = str(components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def resolve_same_module_function(caller_func, called_name):
    """Try to resolve a function name to an actual function in the same module."""
    if called_name in caller_func.__globals__:
        obj = caller_func.__globals__[called_name]
        if callable(obj) and hasattr(obj, '__code__'):
            if hasattr(obj, '__module__') and obj.__module__ == caller_func.__module__:
                return obj
    return None


def transitive_ast_hash(func, visited=None):
    """
    Hash a function and all same-module functions it calls, transitively.
    """
    if visited is None:
        visited = set()
    
    if id(func) in visited:
        return ""
    visited.add(id(func))
    
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        base_hash = ast.dump(tree)
    except Exception:
        return ""
    
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
    
    dep_hashes = [base_hash]
    for name in sorted(called_names):
        called_func = resolve_same_module_function(func, name)
        if called_func:
            dep_hashes.append(transitive_ast_hash(called_func, visited))
    
    return hashlib.sha256("".join(dep_hashes).encode()).hexdigest()[:16]


print("=" * 80)
print("EXPERIMENT A: Global variable value tracking")
print("=" * 80)

THRESHOLD_A = 0.5
THRESHOLD_B = 0.7

def func_with_threshold(x):
    return x > THRESHOLD_A

print(f"\nInitial THRESHOLD_A = {THRESHOLD_A}")
print(f"  func_with_threshold globals hash: {get_function_globals_hash(func_with_threshold)}")
print(f"  comprehensive hash: {comprehensive_function_hash(func_with_threshold)}")

THRESHOLD_A = 0.9
print(f"\nAfter changing THRESHOLD_A = {THRESHOLD_A}")
print(f"  func_with_threshold globals hash: {get_function_globals_hash(func_with_threshold)}")
print(f"  comprehensive hash: {comprehensive_function_hash(func_with_threshold)}")
print("  NOTE: The hash DOES change when we track global values!")


print("\n" + "=" * 80)
print("EXPERIMENT B: Closure value tracking")
print("=" * 80)

def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

mult_2 = make_multiplier(2)
mult_3 = make_multiplier(3)

print(f"\nmult_2 (factor=2):")
print(f"  closure hash: {get_function_closure_hash(mult_2)}")
print(f"  comprehensive hash: {comprehensive_function_hash(mult_2)}")

print(f"\nmult_3 (factor=3):")
print(f"  closure hash: {get_function_closure_hash(mult_3)}")
print(f"  comprehensive hash: {comprehensive_function_hash(mult_3)}")
print("  NOTE: Different closure values produce different hashes!")


print("\n" + "=" * 80)
print("EXPERIMENT C: Default argument tracking")
print("=" * 80)

default_config = {"learning_rate": 0.01}

def train_with_defaults(data, config=default_config):
    return data

print(f"\nInitial default_config = {default_config}")
print(f"  defaults hash: {get_defaults_hash(train_with_defaults)}")
print(f"  comprehensive hash: {comprehensive_function_hash(train_with_defaults)}")

default_config["learning_rate"] = 0.001
print(f"\nAfter mutating default_config = {default_config}")
print(f"  defaults hash: {get_defaults_hash(train_with_defaults)}")
print(f"  comprehensive hash: {comprehensive_function_hash(train_with_defaults)}")
print("  NOTE: The hash DOES change for mutable defaults!")


print("\n" + "=" * 80)
print("EXPERIMENT D: Transitive hashing")
print("=" * 80)

def helper_v1(x):
    return x * 2

def helper_v2(x):
    return x * 3

def main_function(data):
    return helper_v1(data)

print(f"\nWith helper_v1 (x * 2):")
print(f"  main_function transitive hash: {transitive_ast_hash(main_function)}")

original_helper = helper_v1
exec("helper_v1 = lambda x: x * 3", globals())

print(f"\nAfter redefining helper_v1 (x * 3):")
print(f"  main_function transitive hash: {transitive_ast_hash(main_function)}")
print("  NOTE: This WOULD change if we re-resolve and re-hash!")

helper_v1 = original_helper


print("\n" + "=" * 80)
print("EXPERIMENT E: Lazy import extraction with resolution")
print("=" * 80)

def extract_lazy_imports_with_module_hash(func):
    """Extract lazy imports and hash the imported functions."""
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append((node.module, alias.name, alias.asname or alias.name))
    
    import_hashes = []
    for module_name, import_name, local_name in imports:
        if module_name is None:
            continue
        try:
            import importlib
            module = importlib.import_module(module_name)
            imported_obj = getattr(module, import_name, None)
            
            if imported_obj and callable(imported_obj) and hasattr(imported_obj, '__code__'):
                mod_file = getattr(module, '__file__', '')
                if mod_file and 'site-packages' not in mod_file:
                    try:
                        import_source = textwrap.dedent(inspect.getsource(imported_obj))
                        import_hash = hashlib.sha256(import_source.encode()).hexdigest()[:16]
                        import_hashes.append((f"{module_name}.{import_name}", import_hash))
                    except Exception:
                        pass
        except Exception:
            pass
    
    return import_hashes


def func_with_imports():
    from os.path import join, exists
    from collections import Counter
    return Counter()

print(f"\nLazy imports with hashes: {extract_lazy_imports_with_module_hash(func_with_imports)}")
print("  NOTE: Only user-defined functions would have hashes")


print("\n" + "=" * 80)
print("EXPERIMENT F: Pickle-based function comparison")
print("=" * 80)

print("\nTrying to pickle functions for comparison...")

def simple_func(x):
    return x + 1

try:
    pickled = pickle.dumps(simple_func)
    print(f"  simple_func pickle size: {len(pickled)} bytes")
    print(f"  pickle hash: {hashlib.sha256(pickled).hexdigest()[:16]}")
except Exception as e:
    print(f"  ERROR: {e}")

try:
    import cloudpickle
    cp_pickled = cloudpickle.dumps(simple_func)
    print(f"  cloudpickle size: {len(cp_pickled)} bytes")
    print(f"  cloudpickle hash: {hashlib.sha256(cp_pickled).hexdigest()[:16]}")
except ImportError:
    print("  cloudpickle not installed - skipping")
except Exception as e:
    print(f"  cloudpickle ERROR: {e}")


print("\n" + "=" * 80)
print("EXPERIMENT G: co_code stability across Python versions")
print("=" * 80)

import sys
print(f"\nPython version: {sys.version}")

def sample_func(a, b):
    c = a + b
    return c * 2

code = sample_func.__code__
print(f"\nco_code (bytecode): {code.co_code.hex()[:40]}...")
print(f"co_consts: {code.co_consts}")
print(f"co_names: {code.co_names}")
print(f"co_varnames: {code.co_varnames}")

print("""
NOTE: co_code bytecode can differ between:
- Python versions (3.10 vs 3.11 vs 3.12)
- Optimization levels (-O flag)
- Platform (though usually stable)

AST is more stable across versions but doesn't capture
some semantic information that bytecode does.
""")


print("\n" + "=" * 80)
print("SUMMARY: ALTERNATIVE APPROACHES")
print("=" * 80)
print("""
APPROACH 1: Comprehensive Hash (AST + Globals + Closure + Defaults)
  PROS:
    - Catches global variable value changes
    - Catches closure variable value changes
    - Catches default argument mutations
    - More accurate than pure AST
  CONS:
    - Hashing global values can be expensive for large objects
    - May over-trigger if globals are updated but not used
    - Complex objects may not repr() cleanly
  RECOMMENDATION: Use for high-fidelity mode or as opt-in

APPROACH 2: Transitive AST Hashing
  PROS:
    - Automatically tracks same-module helper changes
    - No user annotation needed
    - Works well in practice
  CONS:
    - Can't track cross-module dependencies without imports
    - Name resolution can fail for complex patterns
  RECOMMENDATION: Use as default for same-module dependencies

APPROACH 3: Lazy Import Tracking
  PROS:
    - Makes cross-module deps explicit in code
    - AST can extract and resolve them
    - Natural Python pattern
  CONS:
    - Requires user to follow convention
    - Some users prefer top-level imports
  RECOMMENDATION: Encourage as best practice, provide fallback

APPROACH 4: Cloudpickle-based Hashing
  PROS:
    - Captures entire function closure
    - Most complete representation
    - Works for anonymous functions
  CONS:
    - Heavy dependency (cloudpickle)
    - Slow for large closures
    - Overkill for most use cases
  RECOMMENDATION: Consider for edge cases only

HYBRID RECOMMENDATION:
1. Use AST hashing as the primary mechanism (fast, stable)
2. Add transitive same-module function hashing (catches helpers)
3. Track lazy imports for cross-module deps (requires convention)
4. Optionally include global value hashing (catches global changes)
5. Provide explicit escape hatches for edge cases
6. Document all limitations clearly
""")

