"""Generate API reference pages automatically from docstrings."""

# pyright: reportMissingImports=false

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Public API modules to document (ordered for navigation): (module, title, members or None for all)
PUBLIC_MODULES: list[tuple[str, str, list[str] | None]] = [
    ("pivot.compose", "Pipelines and Stages", ["Pipeline", "stage", "ArtifactHandle"]),
    ("pivot.stage_def", "Stage Parameters", ["StageParams"]),
    ("pivot.loaders", "Loaders", None),
    ("pivot.decorators", "Decorators", ["no_fingerprint", "fingerprint_by_name"]),
]

for module_path, display_name, members in PUBLIC_MODULES:
    parts = module_path.split(".")
    doc_path = "/".join(parts) + ".md"
    full_doc_path = "reference/" + doc_path

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# {display_name}\n\n")
        fd.write(f"::: {module_path}\n")
        if members is not None:
            fd.write("    options:\n      members:\n")
            fd.writelines(f"        - {member}\n" for member in members)

    nav[tuple(parts)] = doc_path

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
