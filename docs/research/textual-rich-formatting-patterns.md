# Textual/Rich Formatting Patterns for Artifact Identity Display

**Research Date:** 2026-02-15  
**Purpose:** Inform task 9 design for displaying artifact identity (base name + variant) in Pivot TUI/CLI

## Executive Summary

This document compiles best practices and patterns from Textual/Rich documentation and real-world codebases for displaying structured information (identities, paths, multi-column data) in terminal UIs.

**Key Findings:**
1. **Rich Text objects** are the primary mechanism for styled, truncatable content in both Rich and Textual
2. **Text.assemble()** and **Text.from_markup()** are the two main construction patterns
3. **Text.truncate()** provides built-in ellipsis handling with overflow control
4. **DataTable** and **ListView** are the primary Textual widgets for structured lists
5. Current Pivot TUI uses manual truncation with string slicing - could leverage Rich's built-in methods

---

## 1. Rich Text Construction Patterns

### 1.1 Text.assemble() - Programmatic Construction

**Use when:** Building text from multiple styled parts programmatically

```python
from rich.text import Text

# Basic assembly with inline styles
content = Text.assemble(
    ("● ", "green"),           # (text, style) tuples
    "Running Opik Evaluation - ",
    (algorithm, "blue"),
    "\n"
)

# Complex multi-part assembly
step_content = Text.assemble()
step_content.append(f"{step.title}\n", "bold")
step_content.append(Text.from_markup(f"[bold]Action:[/bold] {step.action}\n", style="dim"))
```

**Advantages:**
- Type-safe style application
- No markup escaping needed
- Composable - can append to existing Text objects

### 1.2 Text.from_markup() - Declarative Construction

**Use when:** Working with markup strings (BBCode-like syntax)

```python
from rich.text import Text

# Simple markup
title = Text.from_markup("[bold cyan]Success![/]")

# Complex markup with multiple styles
label = Text.from_markup(
    f"[bold]Select credential to edit or type [red]'b'[/red] to go back[/bold]"
)

# Combining with additional styling
text = Text.from_markup(f"[bold]{function}[/]")
text.append_text(Text.from_markup(f"  [dim cyan]{code_position}[/]"))
```

**Advantages:**
- Concise for simple cases
- Familiar BBCode-like syntax
- Easy to read inline

**Caution:** Must escape user content with `rich.markup.escape()` to prevent injection

---

## 2. Text Truncation and Overflow

### 2.1 Text.truncate() Method

**Signature:** `text.truncate(max_width, overflow="ellipsis")`

```python
from rich.text import Text

# Basic truncation with ellipsis
dataset_text = Text(dataset, overflow="ellipsis")
dataset_text.truncate(25)

# Truncation in table cells
title_text.truncate(truncate_width, overflow="ellipsis")

# Multi-line text truncation
model_parts = model.split("/")
model_text = Text("\n".join(model_parts), overflow="ellipsis")
model_text.truncate(20)
```

**Overflow modes:**
- `"ellipsis"` - Add … at truncation point (default)
- `"crop"` - Hard cut at width
- `"fold"` - Wrap to next line

### 2.2 Current Pivot TUI Pattern (Manual)

```python
# From stage_list.py:126-129
display_name = self._base_name
if len(display_name) > name_width:
    display_name = display_name[: name_width - 1] + "…"
name_escaped = rich.markup.escape(display_name)
```

**Improvement opportunity:** Could use `Text.truncate()` for consistent behavior

---

## 3. Textual Widget Patterns

### 3.1 DataTable - Structured Multi-Column Data

**Use when:** Displaying tabular data with sortable columns

```python
from textual.widgets import DataTable
from rich.text import Text

table = self.query_one(DataTable)

# Add columns with styling
table.add_columns("Name", "Status", "Path")

# Add rows with Rich Text objects for styling
for item in items:
    styled_row = [
        Text(str(cell), style="italic #03AC13", justify="right") 
        for cell in row
    ]
    table.add_row(*styled_row)

# Add labeled rows (non-interactive left column)
label = Text(str(number), style="#B0FC38 italic")
table.add_row(*row, label=label)
```

**Key features:**
- Automatic column width management
- Built-in sorting support
- Cell-level styling with Rich Text
- Row labels for identifiers

### 3.2 ListView - Simple Scrollable Lists

**Use when:** Displaying single-column lists with custom item widgets

```python
from textual.widgets import ListView, ListItem

class CustomListView(ListView):
    def compose(self) -> ComposeResult:
        for item in items:
            yield ListItem(CustomWidget(item))
```

**Key features:**
- Simpler than DataTable for single-column lists
- Custom widget composition per item
- Built-in keyboard navigation

### 3.3 Current Pivot TUI Pattern (Custom Widgets)

```python
# From stage_list.py - Custom StageRow widget
class StageRow(textual.widgets.Static):
    def update_display(self, is_selected: bool | None = None):
        symbol, style = status.get_status_symbol(self._info.status, self._info.reason)
        display_name = self._info.name
        name_escaped = rich.markup.escape(display_name)
        # ... manual layout construction
```

**Current approach:** Custom Static widgets with manual layout  
**Alternative:** Could use DataTable for more structured display

---

## 4. Path Display Patterns

### 4.1 File Tree Display

```python
from rich.tree import Tree
from rich.text import Text
from rich.markup import escape

def build_tree(directory, tree):
    for path in sorted(pathlib.Path(directory).iterdir()):
        if path.is_dir():
            branch = tree.add(f"📁 {escape(path.name)}", style="bold magenta")
            build_tree(path, branch)
        else:
            size = decimal(path.stat().st_size)
            tree.add(f"📄 {escape(path.name)} ({size})")
```

### 4.2 Path Truncation Strategies

**Strategy 1: Middle truncation (preserve extension)**
```python
# Not directly supported by Rich - would need custom logic
"very/long/path/to/file.csv" → "very/.../file.csv"
```

**Strategy 2: End truncation (preserve start)**
```python
text = Text("very/long/path/to/file.csv")
text.truncate(20, overflow="ellipsis")
# Result: "very/long/path/to/…"
```

**Strategy 3: Component-based display**
```python
# Show only filename, full path on hover/detail view
display = path.name  # "file.csv"
tooltip = str(path)  # "very/long/path/to/file.csv"
```

---

## 5. Panel and Container Patterns

### 5.1 Rich Panels for Grouped Content

```python
from rich.panel import Panel
from rich import box

# Simple panel with title
console.print(Panel("Content", title="Title", border_style="blue"))

# Panel with subtitle and padding
console.print(
    Panel(
        "[bold cyan]Success![/]\n\nOperation completed.",
        title="✓ Status",
        subtitle="Press any key to continue",
        padding=(1, 2),
        border_style="green"
    )
)

# Fit panel to content
console.print(Panel.fit("Compact panel", border_style="red"))

# Custom box style
console.print(Panel("Content", box=box.DOUBLE, title="Important"))
```

### 5.2 Tables for Structured Display

```python
from rich.table import Table

table = Table(title="Artifacts", show_header=True, header_style="bold magenta")

# Add columns with specific styles and alignment
table.add_column("Identity", style="cyan", no_wrap=True)
table.add_column("Path", style="dim")
table.add_column("Status", justify="right", style="green")

# Add rows with inline markup
table.add_row("train@current", "models/train.pkl", "[green]●[/green] Ready")
table.add_row("train@baseline", "models/baseline.pkl", "[red]✗[/red] Missing")
```

---

## 6. Styling Conventions

### 6.1 Common Color Patterns

From surveyed codebases:

| Element | Style | Example |
|---------|-------|---------|
| Success/Complete | `green`, `green bold` | `[green]●[/green]` |
| Running/Active | `blue`, `blue bold` | `[blue]▶[/blue]` |
| Error/Failed | `red`, `red bold` | `[red]✗[/red]` |
| Warning | `yellow`, `yellow bold` | `[yellow]⚠[/yellow]` |
| Muted/Secondary | `dim`, `grey50` | `[dim]details[/dim]` |
| Emphasis | `bold`, `bold cyan` | `[bold]Important[/bold]` |
| Code/Paths | `cyan`, `dim cyan` | `[cyan]path/to/file[/cyan]` |

### 6.2 Current Pivot TUI Styles

```python
# From stage_list.py
"[blue bold]▶{count}[/]"      # Running
"[green bold]●{count}[/]"     # Completed
"[red bold]!{count}[/]"       # Failed
```

---

## 7. Real-World Examples

### 7.1 Memray Tree Renderer (Path + Metadata)

```python
# From memray tree.py
ret = Text.from_markup(
    ":open_file_folder:" if allow_expand else ":page_facing_up:"
)
ret.append_text(Text(f" {size_str} ", style=Style(color=size_color.rich_color)))

if node.location is not None:
    function, file, lineno = node.location
    code_position = f"{_filename_to_module_name(file)}:{lineno}" if lineno != 0 else file
    ret.append_text(Text.from_markup(f"[bold]{function}[/]"))
    if code_position:
        ret.append_text(Text.from_markup(f"  [dim cyan]{code_position}[/]"))
```

**Pattern:** Icon + primary info (bold) + secondary info (dim)

### 7.2 Textual DataTable with Styled Cells

```python
# From Textual docs
table.add_columns(*ROWS[0])
for row in ROWS[1:]:
    styled_row = [
        Text(str(cell), style="italic #03AC13", justify="right") 
        for cell in row
    ]
    table.add_row(*styled_row)
```

**Pattern:** Convert all cells to Text objects for consistent styling

### 7.3 Opik Optimizer Display (Truncated Paths)

```python
# From opik optimizer
dataset_text = Text(dataset, overflow="ellipsis")
dataset_text.truncate(25)

model_parts = model.split("/")
model_text = Text("\n".join(model_parts), overflow="ellipsis")
model_text.truncate(20)
```

**Pattern:** Split long paths on separators, truncate with ellipsis

---

## 8. Recommendations for Pivot Artifact Display

### 8.1 Identity Display Format

**Option A: Inline format (current style)**
```
train@current
train@baseline
preprocess
```

**Option B: Separated format**
```
train       @current
train       @baseline
preprocess  (no variant)
```

**Option C: Tree format**
```
train
  @current
  @baseline
preprocess
```

### 8.2 Suggested Implementation Pattern

```python
from rich.text import Text

def format_artifact_identity(base_name: str, variant: str | None, max_width: int) -> Text:
    """Format artifact identity with optional variant."""
    if variant:
        # Assemble base + variant with different styles
        text = Text.assemble(
            (base_name, "bold"),
            ("@", "dim"),
            (variant, "cyan")
        )
    else:
        text = Text(base_name, style="bold")
    
    # Truncate if needed
    if max_width > 0:
        text.truncate(max_width, overflow="ellipsis")
    
    return text
```

### 8.3 Path Display Strategy

**For stage list (compact):**
- Show only filename: `train.pkl`
- Full path in detail panel or tooltip

**For detail view (expanded):**
- Show relative path: `models/train.pkl`
- Truncate with ellipsis if > available width
- Consider middle truncation for very long paths

**For diff view:**
- Show full relative path (no truncation)
- Use dim style for unchanged portions
- Highlight changed portions in bold

---

## 9. Testing Considerations

### 9.1 Width Calculation

Current Pivot TUI pattern:
```python
# Calculate available width accounting for prefix/suffix
available_width = self.size.width if self.size.width > 0 else 33
suffix_visible_len = len(count_str) + 2 + sum(...)
name_width = max(1, available_width - len(prefix) - suffix_visible_len - 1)
```

**Note:** Manual width calculation is error-prone. Consider using Rich's built-in layout system.

### 9.2 Resize Handling

```python
def on_resize(self, _event: Resize) -> None:
    """Re-render when resized to maintain proper truncation."""
    self.update_display()
```

**Important:** Truncation must be recalculated on resize events.

---

## 10. References

### Documentation
- [Textual DataTable Widget](https://textual.textualize.io/widgets/data_table)
- [Rich Text API](https://rich.readthedocs.io/en/stable/text.html)
- [Rich Console Overflow](https://rich.readthedocs.io/en/stable/console.html#overflow)

### Real-World Examples
- **Memray** (`memray/reporters/tree.py`) - File tree with metadata
- **Opik Optimizer** (`opik_optimizer/utils/display/terminal.py`) - Truncated paths
- **Textual Examples** (`docs/examples/widgets/data_table_*.py`) - DataTable patterns
- **Pivot TUI** (`pivot_tui/widgets/stage_list.py`) - Current implementation

### Key Patterns
1. Use `Text.assemble()` for programmatic construction
2. Use `Text.from_markup()` for declarative markup
3. Use `Text.truncate()` for consistent ellipsis handling
4. Use DataTable for multi-column structured data
5. Use ListView for simple scrollable lists
6. Always escape user content with `rich.markup.escape()`
7. Recalculate layout on resize events
