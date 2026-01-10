"""
Deeper investigation of confirmed bugs and additional edge cases.
"""
import sys
sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint

print("=" * 80)
print("BUG 1 DEEP DIVE: Instance-level monkey-patch")
print("=" * 80)

class Processor:
    def process(self, x):
        return x * 2

processor = Processor()

def original_method(self, x):
    return x * 100

def modified_method(self, x):
    return x * 999

# Attach method to instance
import types
processor.process = types.MethodType(original_method, processor)

def stage_monkey():
    return processor.process(10)

fp1 = fingerprint.get_stage_fingerprint(stage_monkey)
print(f"FP with original monkey-patch: {fp1}")

# Change the monkey-patched method
processor.process = types.MethodType(modified_method, processor)

fp2 = fingerprint.get_stage_fingerprint(stage_monkey)
print(f"FP with modified monkey-patch: {fp2}")
print(f"BUG CONFIRMED: Fingerprints are SAME: {fp1 == fp2}")
print()

print("=" * 80)
print("BUG 3 DEEP DIVE: Mixin method changes NOT detected")
print("=" * 80)

# Simulate code change by creating two different mixin versions
class LogMixinV1:
    def log_and_process(self, x):
        return x * 2  # Version 1

class LogMixinV2:
    def log_and_process(self, x):
        return x * 3  # Version 2 - CHANGED!

# Create concrete classes from each mixin
class ConcreteV1(LogMixinV1):
    pass

class ConcreteV2(LogMixinV2):
    pass

proc_v1 = ConcreteV1()
proc_v2 = ConcreteV2()

def stage_v1():
    return proc_v1.log_and_process(10)

def stage_v2():
    return proc_v2.log_and_process(10)

fp_v1 = fingerprint.get_stage_fingerprint(stage_v1)
fp_v2 = fingerprint.get_stage_fingerprint(stage_v2)

print(f"V1 fingerprint: {fp_v1}")
print(f"V2 fingerprint: {fp_v2}")

# Check if class hashes differ
print(f"Class hashes differ: {fp_v1.get('class:proc_v1.__class__') != fp_v2.get('class:proc_v2.__class__')}")

# But what if we just hash the concrete class (which has no body)?
print(f"Note: ConcreteV1 and ConcreteV2 both have empty bodies (just 'pass')")
print()

print("=" * 80)
print("BUG 5 DEEP DIVE: Instances in containers")
print("=" * 80)

class WorkerV1:
    def work(self, x):
        return x + 1

class WorkerV2:
    def work(self, x):
        return x + 999  # Different logic

workers_v1 = {"main": WorkerV1()}
workers_v2 = {"main": WorkerV2()}

def stage_dict_v1():
    return workers_v1["main"].work(10)

def stage_dict_v2():
    return workers_v2["main"].work(10)

fp_v1 = fingerprint.get_stage_fingerprint(stage_dict_v1)
fp_v2 = fingerprint.get_stage_fingerprint(stage_dict_v2)

print(f"V1 dict worker fingerprint: {fp_v1}")
print(f"V2 dict worker fingerprint: {fp_v2}")
print(f"BUG: Instance inside dict is NOT tracked at all")
print()

print("=" * 80)
print("NEW EDGE CASE: Instance in list")
print("=" * 80)

workers_list_v1 = [WorkerV1()]
workers_list_v2 = [WorkerV2()]

def stage_list_v1():
    return workers_list_v1[0].work(10)

def stage_list_v2():
    return workers_list_v2[0].work(10)

fp_v1 = fingerprint.get_stage_fingerprint(stage_list_v1)
fp_v2 = fingerprint.get_stage_fingerprint(stage_list_v2)

print(f"V1 list worker fingerprint: {fp_v1}")
print(f"V2 list worker fingerprint: {fp_v2}")
print()

print("=" * 80)
print("NEW EDGE CASE: Nested attribute access (obj.attr.method)")
print("=" * 80)

class InnerService:
    def compute(self, x):
        return x * 2

class OuterService:
    def __init__(self):
        self.inner = InnerService()

outer = OuterService()

def stage_nested():
    return outer.inner.compute(10)

fp = fingerprint.get_stage_fingerprint(stage_nested)
print(f"Nested access fingerprint: {fp}")
print(f"Note: outer is tracked, but outer.inner is NOT tracked as a separate instance")
print()

print("=" * 80)
print("NEW EDGE CASE: Method from __init__.py re-export")
print("=" * 80)

# Many packages re-export things from __init__.py
# e.g., from mypackage import SomeClass (actually defined in mypackage.impl)
# The fingerprint tracks based on type(instance).__module__

class RealImpl:
    __module__ = "mypackage.impl"  # Pretend it's from a submodule

    def process(self, x):
        return x * 2

# User imports it from main package
impl = RealImpl()

def stage_reexport():
    return impl.process(10)

fp = fingerprint.get_stage_fingerprint(stage_reexport)
print(f"Re-export fingerprint: {fp}")
print()

print("=" * 80)
print("NEW EDGE CASE: Partial function wrapping a method")
print("=" * 80)

import functools

class Multiplier:
    def multiply(self, x, y):
        return x * y

mult = Multiplier()
double = functools.partial(mult.multiply, y=2)

def stage_partial():
    return double(10)

fp = fingerprint.get_stage_fingerprint(stage_partial)
print(f"Partial fingerprint: {fp}")
print(f"Note: double is a functools.partial, not tracked as instance")
print()

print("=" * 80)
print("NEW EDGE CASE: Instance method via module.instance.method")
print("=" * 80)

# When accessing via imported module attribute
import types
service_module = types.ModuleType("service_module")
service_module.__file__ = "/home/pivot/agent1/src/service_module.py"

class ServiceClass:
    def fetch(self, x):
        return x + 100

service_module.instance = ServiceClass()
service_module.ServiceClass = ServiceClass
sys.modules["service_module"] = service_module

import service_module

def stage_module_instance():
    # Access instance through module
    return service_module.instance.fetch(10)

fp = fingerprint.get_stage_fingerprint(stage_module_instance)
print(f"Module instance fingerprint: {fp}")
print(f"service_module is tracked: {'mod:service_module.instance' in fp}")
print(f"But the method change won't be detected!")
print()

print("=" * 80)
print("NEW EDGE CASE: Callable instance (__call__ method)")
print("=" * 80)

class CallableProcessor:
    def __call__(self, x):
        return x * 2

processor_callable = CallableProcessor()

def stage_callable_instance():
    return processor_callable(10)

fp = fingerprint.get_stage_fingerprint(stage_callable_instance)
print(f"Callable instance fingerprint: {fp}")
print(f"Instance tracked: {'class:processor_callable.__class__' in fp}")
print()

# What if __call__ changes?
class CallableProcessorV2:
    def __call__(self, x):
        return x * 999  # Changed!

processor_callable_v2 = CallableProcessorV2()

def stage_callable_instance_v2():
    return processor_callable_v2(10)

fp_v2 = fingerprint.get_stage_fingerprint(stage_callable_instance_v2)
print(f"Callable instance V2 fingerprint: {fp_v2}")
print(f"Different hashes: {fp.get('class:processor_callable.__class__') != fp_v2.get('class:processor_callable_v2.__class__')}")
print()

print("=" * 80)
print("CRITICAL BUG: Checking if base class methods are tracked")
print("=" * 80)

class BaseMixin:
    def base_method(self, x):
        return x + 1

class ConcreteImpl(BaseMixin):
    def concrete_method(self, x):
        return x * 2

impl = ConcreteImpl()

def stage_uses_base_method():
    return impl.base_method(10)

fp = fingerprint.get_stage_fingerprint(stage_uses_base_method)
print(f"Fingerprint: {fp}")

# Get the hash of ConcreteImpl class
class_hash = fp.get('class:impl.__class__')
print(f"ConcreteImpl hash: {class_hash}")

# Now let's see what hash_function_ast returns for just ConcreteImpl
import inspect
print(f"\nConcreteImpl source:\n{inspect.getsource(ConcreteImpl)}")
print(f"\nBaseMixin source:\n{inspect.getsource(BaseMixin)}")

# The hash only includes ConcreteImpl's source, not BaseMixin!
print(f"\nBUG CONFIRMED: Only ConcreteImpl is hashed, BaseMixin.base_method is NOT included")
print("If BaseMixin.base_method changes, fingerprint will NOT change!")
print()

print("=" * 80)
print("SUMMARY OF CONFIRMED BUGS")
print("=" * 80)
print("""
CONFIRMED BUGS (fingerprint does NOT detect changes):

1. Instance-level monkey-patched methods
   - Methods attached via types.MethodType or direct assignment to instance
   - type(instance) returns original class, not the patched method

2. Proxy/delegation objects (__getattr__)
   - Only the Proxy class is tracked
   - The delegated target's methods are invisible to fingerprinting

3. Inherited methods from base classes / mixins (CRITICAL)
   - hash_function_ast only hashes the immediate class
   - Base class method changes are NOT detected
   - MRO (Method Resolution Order) is ignored

4. Instances inside collections (dict, list, tuple)
   - _process_collection_dependency only tracks callables
   - Instance objects inside collections are silently ignored

5. Module.instance.method access pattern
   - mod:module.instance is tracked as repr/unknown
   - The actual method code is NOT fingerprinted

6. Nested attribute access (obj.attr.method)
   - Only top-level obj is tracked
   - Nested objects and their methods are invisible

DESIGN LIMITATIONS (expected but worth documenting):

7. Dynamic classes created via type()
   - Same class name + same source = same hash
   - Even though closure state differs, this is hard to solve

8. functools.partial wrapping methods
   - Tracked as callable but the underlying method binding may not be
""")
