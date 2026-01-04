import ast
import hashlib
import inspect
import textwrap
from types import ModuleType


def helper_leaf(x):
    return x * 2


def helper_middle(x):
    return helper_leaf(x) + 1


def helper_top(x):
    return helper_middle(x) + 10


GLOBAL_CONSTANT = 42


def stage_direct_call(data):
    return helper_top(data)


def stage_with_constant(data):
    return data + GLOBAL_CONSTANT


def stage_with_alias(data):
    f = helper_top
    return f(data)


class ModuleAttrExtractor(ast.NodeVisitor):
    def __init__(self):
        self.attrs = []

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.attrs.append((node.value.id, node.attr))
        self.generic_visit(node)


def extract_module_attrs(func):
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    extractor = ModuleAttrExtractor()
    extractor.visit(tree)
    return extractor.attrs


def is_user_code(obj):
    if not hasattr(obj, "__module__"):
        return False
    mod_name = obj.__module__
    if mod_name is None:
        return True
    if mod_name.startswith("_"):
        return False
    if "site-packages" in str(getattr(obj, "__code__", {}).get("co_filename", "")):
        return False
    return True


def get_recursive_fingerprint(func, visited=None):
    if visited is None:
        visited = set()

    if id(func) in visited:
        return {}
    visited.add(id(func))

    manifest = {}

    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_hash = hashlib.md5(ast.dump(tree).encode()).hexdigest()[:8]
        manifest[f"func:{func.__name__}"] = func_hash
    except OSError:
        manifest[f"func:{func.__name__}"] = "no-source"

    cv = inspect.getclosurevars(func)
    all_refs = {**cv.globals, **cv.nonlocals}

    for name, val in all_refs.items():
        if callable(val) and hasattr(val, "__code__"):
            if val.__module__ == func.__module__:
                manifest.update(get_recursive_fingerprint(val, visited))
        elif isinstance(val, (bool, int, float, str, bytes, type(None))):
            manifest[f"const:{name}"] = repr(val)
        elif isinstance(val, ModuleType):
            attrs = extract_module_attrs(func)
            for mod_name, attr_name in attrs:
                if mod_name == name:
                    attr_val = getattr(val, attr_name, None)
                    if callable(attr_val) and hasattr(attr_val, "__code__"):
                        manifest.update(get_recursive_fingerprint(attr_val, visited))

    return manifest


def test_direct_call_captures_transitive():
    print("=" * 60)
    print("TEST 1: Direct call captures transitive dependencies")
    print("=" * 60)

    cv = inspect.getclosurevars(stage_direct_call)
    print(f"Direct globals: {list(cv.globals.keys())}")

    manifest = get_recursive_fingerprint(stage_direct_call)
    print(f"Full manifest: {manifest}")

    expected = {
        "func:stage_direct_call",
        "func:helper_top",
        "func:helper_middle",
        "func:helper_leaf",
    }
    found = set(k for k in manifest.keys() if k.startswith("func:"))

    if expected == found:
        print("SUCCESS: All transitive functions found!")
    else:
        print(f"MISSING: {expected - found}")


def test_global_constant_captured():
    print("\n" + "=" * 60)
    print("TEST 2: Global constants are captured")
    print("=" * 60)

    cv = inspect.getclosurevars(stage_with_constant)
    print(f"Globals: {list(cv.globals.keys())}")

    manifest = get_recursive_fingerprint(stage_with_constant)
    print(f"Manifest: {manifest}")

    if "const:GLOBAL_CONSTANT" in manifest:
        print(f"SUCCESS: Constant captured with value {manifest['const:GLOBAL_CONSTANT']}")
    else:
        print("FAILED: Constant not captured")


def test_aliasing_works():
    print("\n" + "=" * 60)
    print("TEST 3: Aliasing (f = func; f()) works")
    print("=" * 60)

    cv = inspect.getclosurevars(stage_with_alias)
    print(f"Globals: {list(cv.globals.keys())}")

    manifest = get_recursive_fingerprint(stage_with_alias)
    print(f"Manifest: {manifest}")

    if "func:helper_top" in manifest:
        print("SUCCESS: Aliased function captured!")
    else:
        print("FAILED: Aliased function not captured")


def test_module_attr_google_style():
    print("\n" + "=" * 60)
    print("TEST 4: Google-style 'import module' with module.attr")
    print("=" * 60)

    import json

    def stage_with_module(data):
        return json.dumps(data)

    cv = inspect.getclosurevars(stage_with_module)
    print(f"Globals: {list(cv.globals.keys())}")

    attrs = extract_module_attrs(stage_with_module)
    print(f"Module.attr usage: {attrs}")

    for mod_name, attr_name in attrs:
        if mod_name in cv.globals:
            mod = cv.globals[mod_name]
            if isinstance(mod, ModuleType):
                attr = getattr(mod, attr_name, None)
                print(f"Resolved: {mod_name}.{attr_name} -> {type(attr)}")


if __name__ == "__main__":
    test_direct_call_captures_transitive()
    test_global_constant_captured()
    test_aliasing_works()
    test_module_attr_google_style()
