# CHANGELOG

<!-- version list -->

## v0.2.0 (2026-10-15)

### Refactoring

- Compositional pipeline API, path-free engine, and pipeline unification
  ([#452](https://github.com/sjawhar/pivot/pull/452),
  [`0a50002`](https://github.com/sjawhar/pivot/commit/0a500027fa5a02c10c7719418b0314ecff01a9af))

### Breaking Changes

- Pivot.yaml, Pipeline.register, Dep/Out/DirectoryOut annotations, matrix expansion and mutex
  declarations are removed; pipelines are written with @stage and `with Pipeline(...)`. See
  skills/writing-pivot-stages/SKILL.md and docs/migrating-from-dvc.md.


## v0.1.0 (2026-07-13)

- Initial Release
