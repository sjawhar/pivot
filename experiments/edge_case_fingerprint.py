"""
Red team testing: Finding edge cases where instance method changes are NOT detected.

The current fingerprinting mechanism tracks instances via `type(instance)` -> class AST hash.
We need to find scenarios where:
1. An instance exists in closure vars
2. A stage calls a method on it
3. The method code changes
4. But the fingerprint stays the same
"""
import sys
sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint

print("=" * 80)
print("EDGE CASE 1: Method defined on instance, not class")
print("=" * 80)

class BaseProcessor:
    def process(self, x):
        return x * 2

# Create instance and monkey-patch a method directly on the instance
processor1 = BaseProcessor()
processor1.process = lambda self, x: x * 100  # Instance-level override

def stage_uses_processor():
    return processor1.process(10)

fp1 = fingerprint.get_stage_fingerprint(stage_uses_processor)
print(f"Fingerprint keys: {list(fp1.keys())}")
print(f"Class key present: {'class:processor1.__class__' in fp1}")

# Now change the monkey-patched method
processor1.process = lambda self, x: x * 999  # Different logic

fp2 = fingerprint.get_stage_fingerprint(stage_uses_processor)
print(f"Fingerprint changed after monkey-patch: {fp1 != fp2}")
print(f"Fingerprint SAME (BUG!): {fp1 == fp2}")
print()

print("=" * 80)
print("EDGE CASE 2: __getattr__ delegation / Proxy objects")
print("=" * 80)

class RealWorker:
    def compute(self, x):
        return x + 1

class Proxy:
    """Proxy that delegates to another object."""
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)

real_worker_v1 = RealWorker()
proxy = Proxy(real_worker_v1)

def stage_uses_proxy():
    return proxy.compute(10)

fp1 = fingerprint.get_stage_fingerprint(stage_uses_proxy)
print(f"Fingerprint keys: {list(fp1.keys())}")

# The fingerprint tracks Proxy class, but the real work is in RealWorker
# If RealWorker.compute changes, will fingerprint detect it?
print(f"Note: Only Proxy class is tracked, not RealWorker!")
print()

print("=" * 80)
print("EDGE CASE 3: Mixin classes where method is inherited")
print("=" * 80)

class LogMixin:
    def log_and_process(self, x):
        return x * 2

class ConcreteProcessor(LogMixin):
    pass

processor = ConcreteProcessor()

def stage_with_mixin():
    return processor.log_and_process(10)

fp = fingerprint.get_stage_fingerprint(stage_with_mixin)
print(f"Fingerprint keys: {list(fp.keys())}")
print(f"Class tracked: {'class:processor.__class__' in fp}")

# The method is defined in LogMixin, not ConcreteProcessor
# Does the fingerprint include LogMixin's hash?
print(f"Note: log_and_process is from LogMixin, but only ConcreteProcessor is hashed!")
print()

print("=" * 80)
print("EDGE CASE 4: Factory-created instance with closure state")
print("=" * 80)

def create_processor(multiplier):
    class DynamicProcessor:
        def process(self, x):
            return x * multiplier
    return DynamicProcessor()

processor_2x = create_processor(2)
processor_10x = create_processor(10)

def stage_with_2x():
    return processor_2x.process(5)

def stage_with_10x():
    return processor_10x.process(5)

fp_2x = fingerprint.get_stage_fingerprint(stage_with_2x)
fp_10x = fingerprint.get_stage_fingerprint(stage_with_10x)

print(f"2x fingerprint: {fp_2x}")
print(f"10x fingerprint: {fp_10x}")
print(f"Different fingerprints: {fp_2x != fp_10x}")
print(f"SAME fingerprints (BUG!): {fp_2x == fp_10x}")
print()

print("=" * 80)
print("EDGE CASE 5: Instance stored in container (dict/list)")
print("=" * 80)

class Worker:
    def work(self, x):
        return x * 2

workers = {"main": Worker()}

def stage_uses_dict_worker():
    return workers["main"].work(10)

fp = fingerprint.get_stage_fingerprint(stage_uses_dict_worker)
print(f"Fingerprint keys: {list(fp.keys())}")
print(f"Note: workers dict is in closure, but the Worker instance inside is NOT tracked")
print()

print("=" * 80)
print("EDGE CASE 6: Instance imported from another module")
print("=" * 80)

# Simulating: from some_module import shared_instance
# The instance is in globals, but is_user_code might fail if module path is weird

import types
fake_module = types.ModuleType("fake_external")
fake_module.__file__ = "/home/pivot/agent1/src/fake_external.py"

class ExternalService:
    def fetch(self, x):
        return x + 100

fake_module.service = ExternalService()
fake_module.ExternalService = ExternalService
sys.modules["fake_external"] = fake_module

# Now import it
import fake_external

def stage_uses_imported_instance():
    return fake_external.service.fetch(10)

fp = fingerprint.get_stage_fingerprint(stage_uses_imported_instance)
print(f"Fingerprint keys: {list(fp.keys())}")
print(f"service attribute tracked: {'mod:fake_external.service' in fp}")
print()

print("=" * 80)
print("EDGE CASE 7: __class__ changed at runtime")
print("=" * 80)

class OriginalClass:
    def do_work(self):
        return "original"

class ModifiedClass:
    def do_work(self):
        return "modified"

obj = OriginalClass()

def stage_with_mutable_class():
    return obj.do_work()

fp1 = fingerprint.get_stage_fingerprint(stage_with_mutable_class)
print(f"Before class change: {list(fp1.keys())}")

# Now mutate the object's class!
obj.__class__ = ModifiedClass

fp2 = fingerprint.get_stage_fingerprint(stage_with_mutable_class)
print(f"After class change: {list(fp2.keys())}")
print(f"Fingerprint changed: {fp1 != fp2}")
print("Note: This DOES work because type(obj) changes")
print()

print("=" * 80)
print("EDGE CASE 8: Method overridden by descriptor")
print("=" * 80)

class CachedMethod:
    """A descriptor that caches method results but can be replaced."""
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.func.__get__(obj, objtype)

class ServiceWithDescriptor:
    @CachedMethod
    def compute(self, x):
        return x * 2

svc = ServiceWithDescriptor()

def stage_with_descriptor():
    return svc.compute(10)

fp = fingerprint.get_stage_fingerprint(stage_with_descriptor)
print(f"Fingerprint keys: {list(fp.keys())}")
print(f"Note: The compute method is wrapped in a descriptor")
print()

print("=" * 80)
print("EDGE CASE 9: Method from dynamically created class via type()")
print("=" * 80)

def make_processor_class(multiplier):
    def process(self, x):
        return x * multiplier
    return type("DynamicProcessor", (), {"process": process})

DynClass1 = make_processor_class(2)
DynClass2 = make_processor_class(10)

obj1 = DynClass1()
obj2 = DynClass2()

def stage_dyn1():
    return obj1.process(5)

def stage_dyn2():
    return obj2.process(5)

fp1 = fingerprint.get_stage_fingerprint(stage_dyn1)
fp2 = fingerprint.get_stage_fingerprint(stage_dyn2)
print(f"Dynamic class 1 fingerprint keys: {list(fp1.keys())}")
print(f"Dynamic class 2 fingerprint keys: {list(fp2.keys())}")
print(f"Different fingerprints: {fp1 != fp2}")
print(f"SAME fingerprints (potential BUG): {fp1 == fp2}")
print()

print("=" * 80)
print("EDGE CASE 10: Instance method bound at runtime")
print("=" * 80)

class Calculator:
    def add(self, x, y):
        return x + y

    def multiply(self, x, y):
        return x * y

calc = Calculator()
operation = calc.add  # Bound method stored in variable

def stage_with_bound_method():
    return operation(1, 2)

fp1 = fingerprint.get_stage_fingerprint(stage_with_bound_method)
print(f"Fingerprint keys: {list(fp1.keys())}")

# Now swap to multiply
operation = calc.multiply

# Re-fingerprint
fp2 = fingerprint.get_stage_fingerprint(stage_with_bound_method)
print(f"After swapping bound method: {fp1 == fp2}")
print(f"Note: 'operation' is a bound method, closure capture might not update")
print()

print("=" * 80)
print("EDGE CASE 11: Subclass overrides method but instance is typed as parent")
print("=" * 80)

class BaseService:
    def process(self, x):
        return x + 1

class EnhancedService(BaseService):
    def process(self, x):
        return x * 100

# Instance is EnhancedService but often typed/documented as BaseService
service: BaseService = EnhancedService()

def stage_with_subclass():
    return service.process(10)

fp = fingerprint.get_stage_fingerprint(stage_with_subclass)
print(f"Fingerprint keys: {list(fp.keys())}")
print(f"Note: type(service) is EnhancedService, so this should work correctly")
print(f"Class hash would include EnhancedService.process")
print()

print("=" * 80)
print("SUMMARY OF POTENTIAL BUGS")
print("=" * 80)
print("""
1. Instance-level monkey-patched methods NOT detected (methods attached directly to instance)
2. Proxy/delegation patterns - only proxy class tracked, not delegated target
3. Mixin methods - only concrete class hashed, not mixin base classes
4. Factory-created classes with closure state - multiplier not captured
5. Instances inside containers (dict/list) - not recursively tracked
6. Bound methods in closure - swapping doesn't update fingerprint (closure captures early)
7. Dynamic type() classes - may have same name but different implementations
""")
