# pyright: reportUnusedFunction=false, reportUnusedParameter=false
"""
Additional edge cases to test.
"""

import functools
import sys
import types

sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint

print("="*70)
print("ADDITIONAL EDGE CASES")
print("="*70)

# Edge Case 1: operator.methodcaller
print("\n--- Edge Case 1: operator.methodcaller ---")
import operator

caller_v1 = operator.methodcaller("upper")
caller_v2 = operator.methodcaller("lower")  # Different method!

print(f"is_user_code(caller_v1): {fingerprint.is_user_code(caller_v1)}")
print(f"type(caller_v1): {type(caller_v1)}")

# Methodcaller is from stdlib, so not tracked

# Edge Case 2: operator.attrgetter
print("\n--- Edge Case 2: operator.attrgetter ---")

getter_v1 = operator.attrgetter("name")
getter_v2 = operator.attrgetter("value")

print(f"is_user_code(getter_v1): {fingerprint.is_user_code(getter_v1)}")

# Edge Case 3: Methods bound at runtime via __get__
print("\n--- Edge Case 3: Descriptor protocol / __get__ ---")

class Descriptor:
    def __get__(self, obj, objtype=None):
        def method():
            return "from descriptor"
        return method

class MyClass:
    method = Descriptor()

obj1 = MyClass()
obj2 = MyClass()

def stage_descriptor():
    return obj1.method()

fp = fingerprint.get_stage_fingerprint(stage_descriptor)
print(f"Descriptor stage fingerprint: {fp}")
# obj1 is captured, but does it track that method is a descriptor?

# Edge Case 4: __call__ method changes
print("\n--- Edge Case 4: __call__ method ---")

class CallableV1:
    def __call__(self):
        return 1

class CallableV2:
    def __call__(self):
        return 2

callable_instance = CallableV1()

def stage_callable_instance():
    return callable_instance()

fp1 = fingerprint.get_stage_fingerprint(stage_callable_instance)
print(f"CallableV1 fingerprint: {fp1}")

# Now if we change the callable's class (monkey-patch), is it detected?
# This is extreme but possible
callable_instance.__class__ = CallableV2
fp2 = fingerprint.get_stage_fingerprint(stage_callable_instance)
print(f"After class swap fingerprint: {fp2}")
print(f"Detected: {fp1 != fp2}")

# Edge Case 5: Async functions
print("\n--- Edge Case 5: Async functions ---")

async def async_callback_v1():
    return 1

async def async_callback_v2():
    return 2

async_callback = async_callback_v1

async def stage_async():
    return await async_callback()

fp1 = fingerprint.get_stage_fingerprint(stage_async)
print(f"Async stage fingerprint: {fp1}")

async_callback = async_callback_v2
fp2 = fingerprint.get_stage_fingerprint(stage_async)
print(f"After callback change: {fp2}")
print(f"Detected: {fp1 != fp2}")

# Edge Case 6: Generator functions
print("\n--- Edge Case 6: Generator functions ---")

def gen_v1():
    yield 1

def gen_v2():
    yield 2

current_gen = gen_v1

def stage_generator():
    return list(current_gen())

fp1 = fingerprint.get_stage_fingerprint(stage_generator)
print(f"Generator stage fingerprint: {fp1}")

current_gen = gen_v2
fp2 = fingerprint.get_stage_fingerprint(stage_generator)
print(f"After generator change: {fp2}")
print(f"Detected: {fp1 != fp2}")

# Edge Case 7: C extension functions
print("\n--- Edge Case 7: C extension functions (json) ---")
import json

# json.dumps is a C function
def stage_json():
    return json.dumps({"key": "value"})

fp = fingerprint.get_stage_fingerprint(stage_json)
print(f"JSON stage fingerprint: {fp}")
# json module is tracked but as non-user code

# Edge Case 8: Callbacks in dataclass fields
print("\n--- Edge Case 8: Dataclass with callback field ---")
from dataclasses import dataclass

def callback_a():
    return "a"

def callback_b():
    return "b"

@dataclass
class Config:
    callback: types.FunctionType = callback_a  # type: ignore

config = Config()

def stage_dataclass_callback():
    return config.callback()

fp1 = fingerprint.get_stage_fingerprint(stage_dataclass_callback)
print(f"Dataclass callback fingerprint: {fp1}")

config.callback = callback_b  # type: ignore
fp2 = fingerprint.get_stage_fingerprint(stage_dataclass_callback)
print(f"After callback change: {fp2}")
print(f"Detected: {fp1 != fp2}")

# Edge Case 9: Enum with callable values
print("\n--- Edge Case 9: Enum with callable values ---")
import enum

def process_fast():
    return "fast"

def process_slow():
    return "slow"

class ProcessMode(enum.Enum):
    FAST = process_fast
    SLOW = process_slow

mode = ProcessMode.FAST

def stage_enum_callback():
    return mode.value()

fp1 = fingerprint.get_stage_fingerprint(stage_enum_callback)
print(f"Enum callback fingerprint: {fp1}")

mode = ProcessMode.SLOW
fp2 = fingerprint.get_stage_fingerprint(stage_enum_callback)
print(f"After mode change: {fp2}")
print(f"Detected: {fp1 != fp2}")

# Edge Case 10: importlib.import_module at runtime
print("\n--- Edge Case 10: Dynamic imports ---")
import importlib

# If a stage does: mod = importlib.import_module("some_module")
# and then calls mod.function(), that's not tracked

def stage_dynamic_import():
    mod = importlib.import_module("math")
    return mod.sqrt(4)

fp = fingerprint.get_stage_fingerprint(stage_dynamic_import)
print(f"Dynamic import fingerprint: {fp}")
# The 'mod' variable is computed at runtime, not in closure vars

# Edge Case 11: exec'd code
print("\n--- Edge Case 11: exec'd code ---")

exec_code = "def dynamic_func(): return 42"
exec(exec_code)
dynamic_func = eval("dynamic_func")

def stage_execd():
    return dynamic_func()

fp = fingerprint.get_stage_fingerprint(stage_execd)
print(f"Exec'd function fingerprint: {fp}")

# Change the exec'd code
exec_code = "def dynamic_func(): return 999"
exec(exec_code)
dynamic_func_v2 = eval("dynamic_func")

def stage_execd_v2():
    return dynamic_func_v2()

fp2 = fingerprint.get_stage_fingerprint(stage_execd_v2)
print(f"Changed exec'd function fingerprint: {fp2}")

# Edge Case 12: __getattr__ interception
print("\n--- Edge Case 12: __getattr__ interception ---")

class LazyModule:
    """Module that loads callables lazily via __getattr__."""

    def __init__(self):
        self._callbacks = {}

    def register(self, name, callback):
        self._callbacks[name] = callback

    def __getattr__(self, name):
        if name in self._callbacks:
            return self._callbacks[name]
        raise AttributeError(name)

lazy = LazyModule()
lazy.register("process", lambda: 1)

def stage_lazy_attr():
    return lazy.process()

fp = fingerprint.get_stage_fingerprint(stage_lazy_attr)
print(f"Lazy attr fingerprint: {fp}")

lazy.register("process", lambda: 2)
fp2 = fingerprint.get_stage_fingerprint(stage_lazy_attr)
print(f"After callback change: {fp2}")
print(f"Detected: {fp != fp2}")

print("\n" + "="*70)
print("SUMMARY OF ADDITIONAL FINDINGS")
print("="*70)
print("""
Vulnerabilities confirmed:
1. operator.methodcaller/attrgetter - NOT tracked (stdlib)
2. Instance attributes changed after capture - NOT tracked
3. __call__ class changes - MAY be detected (tracks class definition)
4. Module-level callback variables - TRACKED
5. Dataclass instance attributes - NOT tracked
6. Enum values - NOT tracked (enum is the instance, not callable)
7. Dynamic imports - NOT tracked (runtime computed)
8. exec'd code - NOT tracked (no reliable source)
9. __getattr__ intercepted attrs - NOT tracked

The core issue: Fingerprinting captures STATIC closure variables at
function definition time, but many callbacks are resolved DYNAMICALLY
at runtime via attribute access, dictionary lookup, or method resolution.
""")
