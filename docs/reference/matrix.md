# Matrix Stages

Matrix stages let you create multiple variants of the same stage with different configurations.

## YAML Configuration

Matrix stages are defined in `pivot.yaml`:

```yaml
stages:
  train:
    python: stages.train
    deps:
      data: data/clean.csv
      config: "configs/${model}.yaml"
    outs:
      model: "models/${model}_${dataset}.pkl"
    matrix:
      model:
        bert:
          params:
            hidden_size: 768
        gpt:
          params:
            hidden_size: 1024
          deps:
            tokenizer: data/gpt_tokenizer.json
      dataset: [swe, human]
```

This generates 4 stages:

- `train@bert_swe`
- `train@bert_human`
- `train@gpt_swe`
- `train@gpt_human`

## Matrix Expansion Rules

- Each dimension creates variants for all values
- Variants are combined as cross-product
- Variable substitution uses `${variable}` syntax

## Accessing the Variant Name

The stage function receives the variant name via a `variant` parameter if defined:

```python
# stages.py
def train(variant: str):
    print(f"Current variant: {variant}")  # "bert_swe", "bert_human", etc.
```

## Running Matrix Stages

```bash
# Run all variants
pivot run

# Run specific variant
pivot run train@bert_swe

# Run all variants of a base stage
pivot run "train@*"
```

## Listing Matrix Stages

```bash
pivot list
# Output:
# train@bert_swe
# train@bert_human
# train@gpt_swe
# train@gpt_human
```

## Variant Name Rules

Variant names must:

- Contain only alphanumeric characters, underscores, and hyphens
- Be at most 64 characters long
- Be unique within a matrix

```python
# Valid
Variant(name='train_v1')
Variant(name='bert-large')
Variant(name='2024')

# Invalid
Variant(name='')           # Empty
Variant(name='has spaces') # Spaces
Variant(name='has@at')     # @ is reserved
```

## See Also

- [Defining Pipelines](pipelines.md) - Stage definition patterns
- [Parameters](parameters.md) - Parameter handling
