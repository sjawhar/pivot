"""Generate API reference pages automatically from docstrings."""

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Public API modules to document (ordered for navigation)
PUBLIC_MODULES = [
    ("pivot", "Package Exports"),
    ("pivot.registry", "Stage Registration"),
    ("pivot.outputs", "Output Types"),
    ("pivot.pipeline", "Pipeline Class"),
    ("pivot.executor", "Execution"),
    ("pivot.parameters", "Parameters"),
    ("pivot.explain", "Explain Mode"),
    ("pivot.types", "Type Definitions"),
    ("pivot.storage", "Storage & Cache"),
    ("pivot.dvc_compat", "DVC Compatibility"),
]

for module_path, display_name in PUBLIC_MODULES:
    parts = module_path.split(".")
    doc_path = "/".join(parts) + ".md"
    full_doc_path = "reference/" + doc_path

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# {display_name}\n\n")
        fd.write(f"::: {module_path}\n")

    nav[tuple(parts)] = doc_path

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
