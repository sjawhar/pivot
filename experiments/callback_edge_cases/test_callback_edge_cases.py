# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportUnknownLambdaType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
Red team testing for callback/function argument change detection in fingerprinting.

The goal is to find scenarios where:
1. A function is passed as a callback/argument
2. The callback code changes
3. But the fingerprint stays the same (BUG!)
"""

import functools
import sys
import types

# Add src to path for imports
sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint


# =============================================================================
# Scenario 1: Callbacks retrieved from dictionaries at runtime
# =============================================================================

def _callback_v1():
    """Original callback."""
    return 1

def _callback_v2():
    """Changed callback."""
    return 2

CALLBACK_REGISTRY = {
    "process": _callback_v1
}

def stage_uses_dict_callback():
    """Stage that retrieves callback from dict at runtime."""
    callback = CALLBACK_REGISTRY["process"]
    return callback()


def test_dict_callback_detection():
    """Test if callbacks from dict are detected."""
    print("\n=== Scenario 1: Dict-retrieved callbacks ===")

    # Get fingerprint with v1
    CALLBACK_REGISTRY["process"] = _callback_v1
    fp1 = fingerprint.get_stage_fingerprint(stage_uses_dict_callback)

    # Change callback to v2
    CALLBACK_REGISTRY["process"] = _callback_v2
    fp2 = fingerprint.get_stage_fingerprint(stage_uses_dict_callback)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # Check if CALLBACK_REGISTRY is in fingerprint
    has_registry = any("CALLBACK_REGISTRY" in k for k in fp1.keys())
    print(f"CALLBACK_REGISTRY captured: {has_registry}")

    if fp1 == fp2:
        print("VULNERABILITY: Dict callback change NOT detected!")
    else:
        print("OK: Dict callback change detected")

    return fp1 == fp2


# =============================================================================
# Scenario 2: Callbacks via getattr()
# =============================================================================

class Handlers:
    @staticmethod
    def process_v1():
        return 1

    @staticmethod
    def process_v2():
        return 2

def stage_uses_getattr_callback():
    """Stage that retrieves callback via getattr."""
    handler_name = "process_v1"
    callback = getattr(Handlers, handler_name)
    return callback()


def test_getattr_callback_detection():
    """Test if callbacks from getattr are detected."""
    print("\n=== Scenario 2: getattr-retrieved callbacks ===")

    fp = fingerprint.get_stage_fingerprint(stage_uses_getattr_callback)
    print(f"FP keys: {list(fp.keys())}")

    # Check if Handlers class is captured
    has_handlers = any("Handlers" in k for k in fp.keys())
    print(f"Handlers captured: {has_handlers}")

    # The issue: changing Handlers.process_v1 would change the class hash,
    # but the string "process_v1" is just a string in the AST
    print("INFO: getattr with string literal - dynamic dispatch not tracked")

    return not has_handlers


# =============================================================================
# Scenario 3: functools.partial
# =============================================================================

def base_function(multiplier, x):
    """Base function to be partialed."""
    return x * multiplier

partial_v1 = functools.partial(base_function, 2)
partial_v2 = functools.partial(base_function, 3)

def stage_uses_partial():
    """Stage using a partial function."""
    return partial_v1(10)


def test_partial_detection():
    """Test if partial function changes are detected."""
    print("\n=== Scenario 3: functools.partial ===")

    # Create the stage function dynamically to test different partials
    def make_stage(p):
        def stage():
            return p(10)
        return stage

    stage1 = make_stage(partial_v1)
    stage2 = make_stage(partial_v2)

    fp1 = fingerprint.get_stage_fingerprint(stage1)
    fp2 = fingerprint.get_stage_fingerprint(stage2)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # Check if partial is in fingerprint
    has_partial = any("partial" in k.lower() or "p[" in k for k in fp1.keys())
    print(f"Partial captured: {has_partial}")

    if fp1 == fp2:
        print("VULNERABILITY: Partial change NOT detected!")
    else:
        print("OK: Partial change detected")

    return fp1 == fp2


# =============================================================================
# Scenario 4: Callbacks selected via runtime conditions
# =============================================================================

MODE = "fast"

def _slow_process():
    return 1

def _fast_process():
    return 2

def stage_conditional_callback():
    """Stage that selects callback based on runtime condition."""
    if MODE == "fast":
        callback = _fast_process
    else:
        callback = _slow_process
    return callback()


def test_conditional_callback_detection():
    """Test if conditional callback selection is detected."""
    print("\n=== Scenario 4: Runtime conditional selection ===")

    global MODE

    MODE = "fast"
    fp1 = fingerprint.get_stage_fingerprint(stage_conditional_callback)

    MODE = "slow"
    fp2 = fingerprint.get_stage_fingerprint(stage_conditional_callback)

    print(f"FP1 keys: {list(fp1.keys())}")

    # Both _fast_process and _slow_process should be in closure vars
    has_fast = any("_fast_process" in k for k in fp1.keys())
    has_slow = any("_slow_process" in k for k in fp1.keys())
    print(f"_fast_process captured: {has_fast}")
    print(f"_slow_process captured: {has_slow}")

    # The fingerprint SHOULD include both functions since both are referenced
    # But does it?

    if fp1 == fp2:
        print("OK: Both callbacks captured, fingerprint stable (as expected)")
    else:
        print("ISSUE: Fingerprint changes with MODE despite code being same")

    return not (has_fast and has_slow)


# =============================================================================
# Scenario 5: Bound methods
# =============================================================================

class Processor:
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def process(self, x):
        return x * self.multiplier

processor_v1 = Processor(2)
processor_v2 = Processor(3)

def stage_bound_method_v1():
    """Stage using bound method from instance v1."""
    return processor_v1.process(10)


def stage_bound_method_v2():
    """Stage using bound method from instance v2."""
    return processor_v2.process(10)


def test_bound_method_detection():
    """Test if bound method instance changes are detected."""
    print("\n=== Scenario 5: Bound methods ===")

    fp1 = fingerprint.get_stage_fingerprint(stage_bound_method_v1)
    fp2 = fingerprint.get_stage_fingerprint(stage_bound_method_v2)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # Check if processor instances are captured
    has_processor1 = any("processor_v1" in k for k in fp1.keys())
    has_processor2 = any("processor_v2" in k for k in fp2.keys())
    print(f"processor_v1 captured: {has_processor1}")
    print(f"processor_v2 captured: {has_processor2}")

    if fp1 == fp2:
        print("VULNERABILITY: Different bound method instances have same fingerprint!")
    else:
        print("OK: Bound method instance difference detected")

    return fp1 == fp2


# =============================================================================
# Scenario 6: Callbacks passed through multiple layers
# =============================================================================

def _inner_callback():
    return 42

def middle_layer(callback):
    """Middle layer that receives callback."""
    def wrapper():
        return callback() + 1
    return wrapper

# The callback is bound at module load time
wrapped_callback = middle_layer(_inner_callback)

def stage_multi_layer():
    """Stage using multi-layer wrapped callback."""
    return wrapped_callback()


def test_multi_layer_callback():
    """Test if callbacks passed through layers are detected."""
    print("\n=== Scenario 6: Multi-layer callbacks ===")

    fp = fingerprint.get_stage_fingerprint(stage_multi_layer)
    print(f"FP keys: {list(fp.keys())}")

    # Check if wrapped_callback and _inner_callback are captured
    has_wrapped = any("wrapped_callback" in k for k in fp.keys())
    has_inner = any("_inner_callback" in k for k in fp.keys())
    print(f"wrapped_callback captured: {has_wrapped}")
    print(f"_inner_callback captured: {has_inner}")

    # The issue: wrapped_callback is a closure that captured _inner_callback
    # Does fingerprinting recurse into wrapped_callback's closure?

    return not has_inner


# =============================================================================
# Scenario 7: Callbacks from factory functions
# =============================================================================

def callback_factory(version):
    """Factory that creates callbacks."""
    def callback():
        return version
    return callback

factory_callback = callback_factory(1)

def stage_factory_callback():
    """Stage using factory-generated callback."""
    return factory_callback()


def test_factory_callback():
    """Test if factory-generated callbacks are detected."""
    print("\n=== Scenario 7: Factory callbacks ===")

    global factory_callback

    factory_callback = callback_factory(1)
    fp1 = fingerprint.get_stage_fingerprint(stage_factory_callback)

    factory_callback = callback_factory(2)
    fp2 = fingerprint.get_stage_fingerprint(stage_factory_callback)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # Check if factory_callback is captured
    has_callback = any("factory_callback" in k for k in fp1.keys())
    print(f"factory_callback captured: {has_callback}")

    # The issue: Both callbacks have the same code (return version)
    # but different nonlocal 'version' values
    # Does fingerprinting capture the nonlocal?

    if fp1 == fp2:
        print("VULNERABILITY: Factory callback with different closures have same fingerprint!")
    else:
        print("OK: Factory callback closure difference detected")

    return fp1 == fp2


# =============================================================================
# Scenario 8: Class attributes
# =============================================================================

class Config:
    callback = _callback_v1


def stage_class_attr_callback():
    """Stage using callback from class attribute."""
    return Config.callback()


def test_class_attr_callback():
    """Test if class attribute callback changes are detected."""
    print("\n=== Scenario 8: Class attribute callbacks ===")

    Config.callback = _callback_v1
    fp1 = fingerprint.get_stage_fingerprint(stage_class_attr_callback)

    Config.callback = _callback_v2
    fp2 = fingerprint.get_stage_fingerprint(stage_class_attr_callback)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # Check if Config class is captured
    has_config = any("Config" in k for k in fp1.keys())
    print(f"Config captured: {has_config}")

    if fp1 == fp2:
        print("VULNERABILITY: Class attribute callback change NOT detected!")
    else:
        print("OK: Class attribute callback change detected")

    return fp1 == fp2


# =============================================================================
# Scenario 9: functools.wraps decorated functions
# =============================================================================

def original_func():
    """Original function."""
    return 1

@functools.wraps(original_func)
def wrapped_func():
    """This is actually different."""
    return 2


def test_wraps_detection():
    """Test if functools.wraps doesn't hide code changes."""
    print("\n=== Scenario 9: functools.wraps ===")

    fp1 = fingerprint.get_stage_fingerprint(original_func)
    fp2 = fingerprint.get_stage_fingerprint(wrapped_func)

    print(f"FP1: {fp1}")
    print(f"FP2: {fp2}")

    if fp1 == fp2:
        print("VULNERABILITY: @functools.wraps hides code change!")
    else:
        print("OK: @functools.wraps doesn't hide code change")

    return fp1 == fp2


# =============================================================================
# Scenario 10: Callbacks from __getitem__ / subscript access
# =============================================================================

class CallbackContainer:
    def __init__(self):
        self._callbacks = {"default": _callback_v1}

    def __getitem__(self, key):
        return self._callbacks[key]

container = CallbackContainer()

def stage_subscript_callback():
    """Stage that gets callback via subscript."""
    callback = container["default"]
    return callback()


def test_subscript_callback():
    """Test if subscript-accessed callbacks are detected."""
    print("\n=== Scenario 10: Subscript access callbacks ===")

    container._callbacks["default"] = _callback_v1
    fp1 = fingerprint.get_stage_fingerprint(stage_subscript_callback)

    container._callbacks["default"] = _callback_v2
    fp2 = fingerprint.get_stage_fingerprint(stage_subscript_callback)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # Check if container is captured
    has_container = any("container" in k for k in fp1.keys())
    print(f"container captured: {has_container}")

    if fp1 == fp2:
        print("VULNERABILITY: Subscript callback change NOT detected!")
    else:
        print("OK: Subscript callback change detected")

    return fp1 == fp2


# =============================================================================
# Scenario 11: Callbacks from module-level variables that change
# =============================================================================

current_callback = _callback_v1

def stage_module_var_callback():
    """Stage that uses module-level callback variable."""
    return current_callback()


def test_module_var_callback():
    """Test if module-level callback variable changes are detected."""
    print("\n=== Scenario 11: Module variable callbacks ===")

    global current_callback

    current_callback = _callback_v1
    fp1 = fingerprint.get_stage_fingerprint(stage_module_var_callback)

    current_callback = _callback_v2
    fp2 = fingerprint.get_stage_fingerprint(stage_module_var_callback)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # Check if current_callback is in fingerprint
    has_callback = any("current_callback" in k for k in fp1.keys())
    print(f"current_callback captured: {has_callback}")

    if fp1 == fp2:
        print("VULNERABILITY: Module variable callback change NOT detected!")
    else:
        print("OK: Module variable callback change detected")

    return fp1 == fp2


# =============================================================================
# Scenario 12: Callbacks registered via decorator that modifies a global
# =============================================================================

registered_handlers = {}

def register(name):
    """Decorator that registers a handler."""
    def decorator(func):
        registered_handlers[name] = func
        return func
    return decorator

@register("handler")
def registered_handler_v1():
    return 1


def stage_registered_callback():
    """Stage that uses registered callback."""
    callback = registered_handlers["handler"]
    return callback()


def test_registered_callback():
    """Test if decorator-registered callbacks are detected."""
    print("\n=== Scenario 12: Decorator-registered callbacks ===")

    # Register v1
    registered_handlers["handler"] = registered_handler_v1
    fp1 = fingerprint.get_stage_fingerprint(stage_registered_callback)

    # Register v2 (different function)
    registered_handlers["handler"] = _callback_v2
    fp2 = fingerprint.get_stage_fingerprint(stage_registered_callback)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # Check if registered_handlers is captured
    has_handlers = any("registered_handlers" in k for k in fp1.keys())
    print(f"registered_handlers captured: {has_handlers}")

    if fp1 == fp2:
        print("VULNERABILITY: Registered callback change NOT detected!")
    else:
        print("OK: Registered callback change detected")

    return fp1 == fp2


# =============================================================================
# Scenario 13: Lambdas with captured variables
# =============================================================================

multiplier = 2
lambda_callback = lambda x: x * multiplier  # noqa: E731

def stage_lambda_with_closure():
    """Stage using lambda with captured variable."""
    return lambda_callback(10)


def test_lambda_closure_detection():
    """Test if lambda closure variable changes are detected."""
    print("\n=== Scenario 13: Lambda with closure ===")

    global lambda_callback, multiplier

    multiplier = 2
    lambda_callback = lambda x: x * multiplier  # noqa: E731
    fp1 = fingerprint.get_stage_fingerprint(stage_lambda_with_closure)

    multiplier = 3
    lambda_callback = lambda x: x * multiplier  # noqa: E731
    fp2 = fingerprint.get_stage_fingerprint(stage_lambda_with_closure)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # The lambda code is the same, but the captured 'multiplier' is different
    # Does fingerprinting detect this?

    has_multiplier = any("multiplier" in k for k in fp1.keys())
    print(f"multiplier captured: {has_multiplier}")

    if fp1 == fp2:
        print("VULNERABILITY: Lambda closure change NOT detected!")
    else:
        print("OK: Lambda closure change detected")

    return fp1 == fp2


# =============================================================================
# Scenario 14: Method resolution order / inheritance
# =============================================================================

class BaseProcessor:
    def process(self):
        return 1

class DerivedProcessorV1(BaseProcessor):
    def process(self):
        return 2

class DerivedProcessorV2(BaseProcessor):
    def process(self):
        return 3

processor_instance = DerivedProcessorV1()

def stage_polymorphic_callback():
    """Stage using polymorphic method."""
    return processor_instance.process()


def test_polymorphic_callback():
    """Test if polymorphic method changes are detected."""
    print("\n=== Scenario 14: Polymorphic callbacks ===")

    global processor_instance

    processor_instance = DerivedProcessorV1()
    fp1 = fingerprint.get_stage_fingerprint(stage_polymorphic_callback)

    processor_instance = DerivedProcessorV2()
    fp2 = fingerprint.get_stage_fingerprint(stage_polymorphic_callback)

    print(f"FP1 keys: {list(fp1.keys())}")
    print(f"FP2 keys: {list(fp2.keys())}")

    # Check if processor_instance is captured
    has_instance = any("processor_instance" in k for k in fp1.keys())
    print(f"processor_instance captured: {has_instance}")

    if fp1 == fp2:
        print("VULNERABILITY: Polymorphic callback change NOT detected!")
    else:
        print("OK: Polymorphic callback change detected")

    return fp1 == fp2


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    vulnerabilities = []

    tests = [
        ("Dict callbacks", test_dict_callback_detection),
        ("getattr callbacks", test_getattr_callback_detection),
        ("functools.partial", test_partial_detection),
        ("Conditional selection", test_conditional_callback_detection),
        ("Bound methods", test_bound_method_detection),
        ("Multi-layer callbacks", test_multi_layer_callback),
        ("Factory callbacks", test_factory_callback),
        ("Class attribute callbacks", test_class_attr_callback),
        ("functools.wraps", test_wraps_detection),
        ("Subscript callbacks", test_subscript_callback),
        ("Module variable callbacks", test_module_var_callback),
        ("Registered callbacks", test_registered_callback),
        ("Lambda closures", test_lambda_closure_detection),
        ("Polymorphic callbacks", test_polymorphic_callback),
    ]

    for name, test in tests:
        try:
            has_vuln = test()
            if has_vuln:
                vulnerabilities.append(name)
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if vulnerabilities:
        print(f"\nVULNERABILITIES FOUND ({len(vulnerabilities)}):")
        for v in vulnerabilities:
            print(f"  - {v}")
    else:
        print("\nNo vulnerabilities found!")
