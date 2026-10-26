# pivot-tui

Interactive terminal UI for [Pivot](https://pypi.org/project/pivot/), the
high-performance Python pipeline tool with automatic code change detection.

Built on [Textual](https://textual.textualize.io/), pivot-tui gives you a live
view of your pipeline as it runs: stage status, dependency/output diffs, logs,
and run history — all updating in real time over Pivot's RPC interface.

## Installation

```bash
pip install pivot[tui]
```

## Usage

```bash
pivot repro --tui           # Run the pipeline with the interactive TUI
pivot repro --tui --watch   # Watch mode: re-runs stages as files change
```

## Learn More

- [Documentation](https://sjawhar.github.io/pivot/)
- [Source code](https://github.com/sjawhar/pivot)
