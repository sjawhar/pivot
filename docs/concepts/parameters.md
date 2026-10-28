# Parameters

Stage parameters are the knobs you turn when experimenting: thresholds,
learning rates, feature flags. Pivot records them in each stage's lock file, so
changing one re-runs that stage, just like changing its code or input data.

## One `StageParams` class per stage

Subclass `StageParams` (a Pydantic `BaseModel`) directly above the stage that
uses it, and pass an instance at the call site:

```python
# pipeline.py
import pandas as pd

from pivot import Pipeline, StageParams, stage


class FilterParams(StageParams):
    min_value: int = 0
    columns: list[str] = ["id", "value"]


@stage
def keep_large(raw: pd.DataFrame, params: FilterParams) -> pd.DataFrame:
    return raw.loc[raw["value"] >= params.min_value, params.columns]


class ScaleParams(StageParams):
    factor: float = 1.0


@stage
def scale(rows: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:
    return rows.assign(value=rows["value"] * params.factor)


with Pipeline("params") as pipeline:
    large = keep_large(pipeline.input("values.csv"), params=FilterParams(min_value=15))
    scale(large, params=ScaleParams(factor=2.0))
```

Each stage owns its parameter class. Do not collect every stage's settings in a
shared `params.py` or config module: a stage then depends on settings it never
reads, and the classes drift away from the code that uses them.

- The parameter name does not matter; Pivot finds it by its `StageParams` type.
  A stage takes at most one, and its annotation must not be a union or `Optional`.
- Mutable defaults are safe (`columns: list[str] = ["id", "value"]`): Pydantic
  copies them. Nested Pydantic models work too.
- Unknown fields are refused, in code (`FilterParams(min_valu=15)`) and in
  `params.yaml` alike. A class that needs extra fields sets its own
  `model_config = pydantic.ConfigDict(extra="allow")`. In `params.yaml` this
  holds at every depth: a typo inside a nested model is refused too, unless that
  model sets `extra="allow"`.
- A stage whose fields all have defaults may declare a default
  (`params: FilterParams = FilterParams()`) and be called without it.
- Values that are not data (a number, a string, a list of names) go here, never
  in a plain function argument: plain arguments are rejected with
  `parameter 'factor' has unsupported type 'int'`.

## Overriding via `params.yaml`

`params.yaml` at the project root overrides values without editing code. Keys
are qualified stage names:

```yaml
params/keep_large:
  min_value: 25
params/scale:
  factor: 10.0
```

Only the listed fields change. For [variants](pipelines.md#variants), a key
without the `@` suffix applies to every variant, and a key with the full name
(`params/scale@big`) is applied on top of it. A field with a Pydantic alias
(`learning_rate: float = Field(0.01, alias="lr")`) takes either name; one key
that sets it under both is refused.

Keys are checked when overrides load. A key with no pipeline part (`scale:`)
is refused with the qualified names it could mean, and so is a key for a
loaded pipeline that names none of its stages (`params/scael:`). Each value
must map parameter names to values, and each stage's params class must accept
them as that stage sees them (base key merged under the variant's key): a
field the class does not have (`factr:`) is refused with the fields it could
mean, and a value its field rejects (`factor: ten`) is refused naming the field,
the value and the reason. Every problem is listed in one error, and no stage
runs. A key for a pipeline that this run does not load (running one pipeline of
several) is skipped.

## Change Detection

Pivot serializes the effective parameters (code values plus `params.yaml`
overrides) into the stage's lock file. On the next run, any difference marks
the stage stale with reason `Params changed`:

- Changing a value in code or in `params.yaml` re-runs the stage
- Changing the class definition (a field's type or default) is caught by
  [fingerprinting](fingerprinting.md) of the class and its schema

## Inspecting Parameters

```bash
pivot params show                # effective values for all stages
pivot params show params/scale   # one stage
pivot params diff                # compare against git HEAD
pivot status --explain           # shows old and new values, including 0 and false, for changed params
```

## Relationship to Other Concepts

| Input type | Tracked by | Triggers re-run when |
|------------|------------|----------------------|
| Parameters | Lock file params dict | Value changes |
| Code | [Fingerprint](fingerprinting.md) manifest | Code changes |
| Data | Dependency hash | File content changes |

A stage is skipped only when all three match its last successful run
([caching](caching.md)).

The lock file lists a `set` or `frozenset` parameter's members in sorted order, so the
recorded value does not depend on Python's hash seed.

### Secret parameters

A secret (`pydantic.SecretStr`, `SecretBytes`, `Secret[T]`) is not recorded as its
value. The lock file holds `hmac:` plus the first 16 hex digits of the value's
HMAC-SHA256 under the project's secret key, so changing the secret re-runs the stage,
while someone who has the lock file but not the key cannot test guesses at the value.
`pivot params show` and `status --explain` show the digest, and an error about an invalid
secret value shows `'**********'`. The one exception is your own code: a
`field_serializer` or `model_serializer` that calls `get_secret_value()` puts what it
returns into the record as written.

The same digest is what the code fingerprint hashes for a secret found outside a stage's
own params: a `StageParams` default on a helper, a loader's config field, or a
`functools.partial` argument, including one inside a dataclass or pydantic model held
there. Changing that secret re-runs the stage, and no manifest or lock holds its value.

The key is `<state dir>/secret.key`: 32 random bytes, created the first time a stage with a
secret is checked, with mode 0600, and listed in the state directory's `.gitignore`. `core.state_dir`
defaults to `.pivot`. Never commit it. Set `PIVOT_SECRET_KEY` (64 hex digits) to use a key from
the environment instead, so CI and teammates record the same digests. A clone without the key,
and without `PIVOT_SECRET_KEY`, makes its own key; each stage that has a secret then re-runs once
there.
