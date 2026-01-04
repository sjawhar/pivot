#!/usr/bin/env python3
"""
Experiments to test various Python function change detection approaches.
This script explores the trade-offs between AST hashing, bytecode hashing,
and source code hashing approaches.
"""
import ast
import dis
import hashlib
import inspect
import sys
import textwrap
import types
from io import StringIO


def hash_function_source(func):
    """Simple source code hashing."""
    source = inspect.getsource(func)
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def hash_function_ast(func):
    """AST-based hashing (normalized)."""
    source = inspect.getsource(func)
    tree = ast.parse(source)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


class DocstringRemover(ast.NodeTransformer):
    """Remove docstrings from AST."""
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
        return node
    
    visit_AsyncFunctionDef = visit_FunctionDef


def hash_function_ast_normalized(func):
    """AST-based hashing with docstring removal."""
    source = inspect.getsource(func)
    tree = ast.parse(source)
    tree = DocstringRemover().visit(tree)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def hash_function_bytecode(func):
    """Bytecode-based hashing using __code__ attributes."""
    code = func.__code__
    parts = [
        code.co_code,
        str(code.co_consts).encode(),
        str(code.co_names).encode(),
        str(code.co_varnames).encode(),
    ]
    combined = b''.join(p if isinstance(p, bytes) else p for p in parts)
    return hashlib.sha256(combined).hexdigest()[:16]


def hash_function_bytecode_full(func):
    """More complete bytecode hashing including free variables."""
    code = func.__code__
    parts = [
        code.co_code,
        str(code.co_consts).encode(),
        str(code.co_names).encode(),
        str(code.co_varnames).encode(),
        str(code.co_freevars).encode(),
        str(code.co_cellvars).encode(),
    ]
    combined = b''.join(p if isinstance(p, bytes) else p for p in parts)
    return hashlib.sha256(combined).hexdigest()[:16]


def extract_lazy_imports(func):
    """Extract lazy imports from function body."""
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append((node.module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, None))
    
    return imports


def extract_called_functions(func):
    """Extract function calls from function body."""
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    calls = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    
    return calls


def extract_name_references(func):
    """Extract all Name references (potential globals/closures)."""
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    
    func_def = tree.body[0]
    if not isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    
    local_names = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_names.add(node.id)
        if isinstance(node, ast.arg):
            local_names.add(node.arg)
    
    referenced_names = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in local_names:
                referenced_names.add(node.id)
    
    return list(referenced_names)


print("=" * 80)
print("EXPERIMENT 1: Basic function hashing comparison")
print("=" * 80)

def example_func_v1(x, y):
    """A simple function."""
    return x + y

def example_func_v2(x, y):
    """A simple function with different docstring."""
    return x + y

def example_func_v3(x, y):
    return x + y

def example_func_v4(a, b):
    return a + b

print("\nFunctions with same logic but different names/docstrings:")
print(f"  v1 (with docstring):           source={hash_function_source(example_func_v1)}")
print(f"                                  ast={hash_function_ast(example_func_v1)}")
print(f"                                  ast_norm={hash_function_ast_normalized(example_func_v1)}")
print(f"                                  bytecode={hash_function_bytecode(example_func_v1)}")

print(f"  v2 (different docstring):      source={hash_function_source(example_func_v2)}")
print(f"                                  ast={hash_function_ast(example_func_v2)}")
print(f"                                  ast_norm={hash_function_ast_normalized(example_func_v2)}")
print(f"                                  bytecode={hash_function_bytecode(example_func_v2)}")

print(f"  v3 (no docstring):             source={hash_function_source(example_func_v3)}")
print(f"                                  ast={hash_function_ast(example_func_v3)}")
print(f"                                  ast_norm={hash_function_ast_normalized(example_func_v3)}")
print(f"                                  bytecode={hash_function_bytecode(example_func_v3)}")

print(f"  v4 (different param names):    source={hash_function_source(example_func_v4)}")
print(f"                                  ast={hash_function_ast(example_func_v4)}")
print(f"                                  ast_norm={hash_function_ast_normalized(example_func_v4)}")
print(f"                                  bytecode={hash_function_bytecode(example_func_v4)}")


print("\n" + "=" * 80)
print("EXPERIMENT 2: Closure and global variable detection")
print("=" * 80)

GLOBAL_THRESHOLD = 0.5
GLOBAL_CONFIG = {"key": "value"}

def func_uses_global(x):
    return x > GLOBAL_THRESHOLD

def func_uses_global_dict(x):
    return x in GLOBAL_CONFIG

def make_closure(multiplier):
    def inner(x):
        return x * multiplier
    return inner

closure_2 = make_closure(2)
closure_3 = make_closure(3)

print("\nGlobal variable detection:")
print(f"  func_uses_global references: {extract_name_references(func_uses_global)}")
print(f"  func_uses_global_dict references: {extract_name_references(func_uses_global_dict)}")

print("\nClosure comparison (same code, different closure values):")
print(f"  closure_2: bytecode={hash_function_bytecode(closure_2)}")
print(f"  closure_3: bytecode={hash_function_bytecode(closure_3)}")
print(f"  closure_2: bytecode_full={hash_function_bytecode_full(closure_2)}")
print(f"  closure_3: bytecode_full={hash_function_bytecode_full(closure_3)}")

print("\n  NOTE: Closures have the SAME bytecode hash! The actual values are in __closure__")
print(f"  closure_2.__closure__: {closure_2.__closure__}")
print(f"  closure_3.__closure__: {closure_3.__closure__}")
print(f"  closure_2.__closure__[0].cell_contents: {closure_2.__closure__[0].cell_contents}")
print(f"  closure_3.__closure__[0].cell_contents: {closure_3.__closure__[0].cell_contents}")


print("\n" + "=" * 80)
print("EXPERIMENT 3: Lazy import extraction")
print("=" * 80)

def func_with_lazy_imports(x):
    from os.path import join
    import json
    from collections import Counter
    
    return Counter(x)

print(f"\nLazy imports in func_with_lazy_imports: {extract_lazy_imports(func_with_lazy_imports)}")


print("\n" + "=" * 80)
print("EXPERIMENT 4: Function call extraction")
print("=" * 80)

def helper_a(x):
    return x * 2

def helper_b(x):
    return x + 1

def main_func(data):
    result = helper_a(data)
    result = helper_b(result)
    return result

print(f"\nFunction calls in main_func: {extract_called_functions(main_func)}")


print("\n" + "=" * 80)
print("EXPERIMENT 5: Edge cases and limitations")
print("=" * 80)

print("\n5a. Dynamic function calls (method calls, getattr):")

class MyProcessor:
    def process(self, x):
        return x * 2

processor = MyProcessor()

def func_with_method_call(x):
    return processor.process(x)

def func_with_dynamic_call(x, method_name="process"):
    return getattr(processor, method_name)(x)

print(f"  Method call extraction: {extract_called_functions(func_with_method_call)}")
print(f"  NOTE: 'process' is extracted, but we can't know it's on MyProcessor")

print(f"\n  Dynamic call extraction: {extract_called_functions(func_with_dynamic_call)}")
print(f"  NOTE: getattr calls are opaque - we can't statically determine what's called")


print("\n5b. Functions defined at different indentation levels:")

def outer_func(x):
    def inner_func(y):
        return y * 2
    return inner_func(x)

print(f"  outer_func source hash: {hash_function_source(outer_func)}")
print(f"  NOTE: inspect.getsource gets the whole function including inner")


print("\n5c. Lambda functions:")

lambda_func = lambda x: x * 2

try:
    print(f"  lambda source: {inspect.getsource(lambda_func)}")
    print(f"  lambda hash: {hash_function_source(lambda_func)}")
except Exception as e:
    print(f"  ERROR getting lambda source: {e}")


print("\n5d. Decorated functions:")

def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def decorated_func(x):
    return x * 2

print(f"  decorated_func.__name__: {decorated_func.__name__}")
print(f"  decorated_func.__wrapped__ exists: {hasattr(decorated_func, '__wrapped__')}")

try:
    print(f"  decorated source: {inspect.getsource(decorated_func)[:50]}...")
except Exception as e:
    print(f"  ERROR: {e}")


print("\n5e. Default argument values:")

default_list = [1, 2, 3]

def func_with_default(x, items=default_list):
    return x in items

print(f"  func_with_default references: {extract_name_references(func_with_default)}")
print(f"  NOTE: default_list is NOT detected as a reference in function body")
print(f"  But it's in __defaults__: {func_with_default.__defaults__}")


print("\n5f. Class methods and self:")

class MyClass:
    def __init__(self):
        self.value = 10
    
    def method(self, x):
        return x + self.value

obj = MyClass()
print(f"  MyClass.method references: {extract_name_references(MyClass.method)}")
print(f"  NOTE: 'self' is detected, but self.value cannot be tracked statically")


print("\n" + "=" * 80)
print("EXPERIMENT 6: inspect.getsource limitations")
print("=" * 80)

print("\n6a. Functions in __main__ vs modules:")
print(f"  All functions here are in __main__, source works fine")

print("\n6b. Dynamically created functions:")

exec_code = """
def dynamic_func(x):
    return x * 2
"""
exec(exec_code)

try:
    print(f"  dynamic_func source: {inspect.getsource(dynamic_func)}")
except Exception as e:
    print(f"  ERROR: Cannot get source of exec'd function: {e}")


print("\n6c. C extension functions:")

try:
    print(f"  len source: {inspect.getsource(len)}")
except TypeError as e:
    print(f"  ERROR: {e}")


print("\n" + "=" * 80)
print("EXPERIMENT 7: Performance measurement")
print("=" * 80)

import time

def complex_func(data, threshold=0.5, multiplier=2):
    """A more complex function for timing tests."""
    import math
    results = []
    for item in data:
        if item > threshold:
            results.append(math.sin(item) * multiplier)
        else:
            results.append(math.cos(item) * multiplier)
    return results


n_iterations = 1000

start = time.perf_counter()
for _ in range(n_iterations):
    hash_function_source(complex_func)
source_time = time.perf_counter() - start

start = time.perf_counter()
for _ in range(n_iterations):
    hash_function_ast(complex_func)
ast_time = time.perf_counter() - start

start = time.perf_counter()
for _ in range(n_iterations):
    hash_function_ast_normalized(complex_func)
ast_norm_time = time.perf_counter() - start

start = time.perf_counter()
for _ in range(n_iterations):
    hash_function_bytecode(complex_func)
bytecode_time = time.perf_counter() - start

print(f"\nTime for {n_iterations} iterations:")
print(f"  Source hashing:       {source_time*1000:.2f} ms ({source_time/n_iterations*1000:.4f} ms/iter)")
print(f"  AST hashing:          {ast_time*1000:.2f} ms ({ast_time/n_iterations*1000:.4f} ms/iter)")
print(f"  AST normalized:       {ast_norm_time*1000:.2f} ms ({ast_norm_time/n_iterations*1000:.4f} ms/iter)")
print(f"  Bytecode hashing:     {bytecode_time*1000:.2f} ms ({bytecode_time/n_iterations*1000:.4f} ms/iter)")


print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print("""
1. AST HASHING PROS:
   - Ignores comments and whitespace (by design)
   - Can be normalized to ignore docstrings
   - Works well for same-module helper functions
   - Captures the semantic structure of code

2. AST HASHING CONS:
   - Different variable names produce different hashes (may be desired)
   - Cannot detect:
     - Global variable value changes
     - Closure value changes
     - Method call targets (obj.method())
     - Dynamic function calls (getattr, exec, eval)
     - Default argument value changes (need to check __defaults__)

3. BYTECODE HASHING PROS:
   - Faster than AST parsing
   - Captures compiled behavior
   - Works for functions without source

4. BYTECODE HASHING CONS:
   - Platform/Python-version specific
   - Same limitations as AST for closures/globals
   - Harder to understand differences

5. CRITICAL EDGE CASES:
   - Closures: Same bytecode, different values
   - Globals: Not detected unless explicitly tracked
   - Default args: Values in __defaults__, not in AST body
   - Method calls: Can't resolve obj.method() statically
   - Decorators: May wrap function, hiding original

6. RECOMMENDED APPROACH:
   - AST hashing as primary mechanism
   - Transitive hashing for same-module functions
   - Lazy import tracking for cross-module
   - Explicit escape hatches for edge cases
   - Document limitations clearly
""")

