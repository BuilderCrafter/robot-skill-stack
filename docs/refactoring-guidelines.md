# Thesis prototype cleanup guidelines

Robot Skill Stack is a master's thesis prototype for controlled Isaac Sim
experiments and a university demonstration. The goal is readable, compact,
maintainable scientific Python that the thesis author can understand. Production
robustness, arbitrary input support and exhaustive regression coverage are not
project goals. No real-robot deployment is in scope.

Read [AGENTS.md](../AGENTS.md) for priorities, the supported environment and essential
experimental correctness rules. This branch starts directly from validated
`origin/main` at `5b61d4442d74e3afd76420ffc11e8e3ab0b21937`.
The unsuccessful `refactor/post-integration-cleanup` is not a source of code,
checker setup or requirements for this work. Preserve the research archive on
`feature/rgbd-rrt-autonomous`; do not merge the old cleanup or rewrite history.

## What a successful cleanup looks like

Prefer direct Python implementations with visible algorithms and little ceremony.
Delete dead code, historical compatibility, unused options, trivial wrappers,
duplicate conversions and unnecessary state. Inline one-use helpers when that
makes the caller clearer; keep helpers for meaningful mathematics or substantial
duplication. Avoid speculative extensibility and design patterns without a real
problem to solve. Do not replace removed code with new abstraction layers.

There is no arbitrary line-count target, but substantial growth during a
simplification refactor is a strong warning. Explain a compelling improvement or
simplify the approach. If a module is already reasonably simple, leave it alone.
A clear 30-line algorithm should not become 80 lines of validation and tracing.

## Minimal, purposeful validation

Evaluate checks individually. Keep a check when removing it would compromise a
supported experiment's result or cause unintended motion. Examples include wrong
observation time/generation, incorrect joint indexing, a colliding trajectory,
motion after cancellation or silent data/source substitution.

Question and remove checks that only support hypothetical or unwanted scenarios:

- Repeated `isinstance()` checks and duplicate numerical validation.
- Revalidation of trusted internal objects and impossible internal states.
- Guards for configurations or input forms we do not intend to support.
- Defensive `getattr()`/`hasattr()` for required interfaces.
- Exception translation, fallback and recovery paths without practical value.

Fail fast. A normal exception and traceback for a programming mistake or unsupported
input is acceptable. Narrower input support and simpler recovery are intentional
tradeoffs. Do not add an error-handling framework to make every failure friendly.
Do not automatically retain a check merely because it is described as defensive
or related to safety; identify its concrete role in the intended experiment.

Keep the intended mathematics and experimental meaning correct. Do not silently
change data, sources, algorithms or meaningful motion constraints to obtain a
passing result. Explain deliberate changes to supported behavior. Cleanup is not
a license to disguise a failed experiment.

## Practical typing, names and documentation

Use annotations where they clarify parameters, results and important state:

```python
def update(self, frames: Sequence[DepthFrame]) -> bool:
    ...
```

Prefer existing concrete classes, simple unions, standard collections and practical
NumPy annotations. Temporary dictionaries and some dynamic typing are acceptable.
Do not introduce large `TypedDict` collections, numerous internal protocols,
one-use abstract interfaces, complex generics or layers of casts/suppressions to
satisfy a checker. Comprehensive type coverage is not the objective. Pyright may
be introduced later as optional development tooling; strict compliance is not a
requirement for each refactor. Do not transfer the old `pyrightconfig.json`,
`requirements-dev.txt` or typing structures.

Choose names that explain the algorithm without turning identifiers into long
sentences. Add short purpose docstrings to nontrivial functions. Comments should
explain mathematics, coordinate conventions or an unusual decision. Remove
historical commentary and narration of obvious statements.

## Diagnostics and tests are cleanup targets

Question timing distributions, diagnostic dictionaries, counters, metadata,
execution traces, debugging flags and repeated serialization. Retain information
actively useful for current experiments, failures and results. An old test is not
an independent consumer that makes obsolete instrumentation necessary. Delete
obsolete tests together with the instrumentation they preserve.

Aggressively remove redundant tests, private implementation assertions, hypothetical
edge cases, historical research expectations, retired-feature tests, defensive
validation tests and oversized low-value harnesses. Keep representative coverage
of core mathematics, important algorithms, working manipulation/planning,
camera/mapping integration, major lifecycle behavior and known important bugs.
A significant decrease in test count is acceptable. Do not replace meaningful
checks with trivial assertions or conceal a real regression by weakening its
expected result. Historical test totals and fixture checks are not blanket
retention requirements for obsolete functionality.

## Testing proportional to the change

| Change | Default validation |
| --- | --- |
| Documentation only | Review the diff and run `git diff --check`; no runtime tests. |
| Typing or naming | Static checking or a small focused test when useful. |
| Local algorithm simplification | Representative tests for that algorithm. |
| Shared runtime behavior | Focused integration tests for the affected behavior. |
| Actual robot motion | An appropriate bounded native Isaac regression. |

Choose the smallest useful set before testing. Do not repeatedly run the complete
offline suite. If a test fails, investigate and rerun the relevant test. A full
regression run is a deliberate milestone justified by the change or explicitly
requested by the user. Do not launch broad native matrices or historical benchmark
batches by default. Use one Isaac instance at a time when native testing is needed.
Inspect relevant results and actual motion when testing execution; process exit
alone is insufficient. Existing scripts and validation reports are tools and
context, not an instruction to run every available gate.

## One module per authorized task

For each separately authorized module:

1. Read its code and actual application callers.
2. Identify obsolete and unnecessary functionality.
3. Remove dead code and low-value abstractions.
4. Remove excessive diagnostics and redundant validation.
5. Simplify state, interfaces and intermediate representations.
6. Add practical typing and improve unclear names.
7. Add short docstrings where useful.
8. Remove redundant or obsolete tests.
9. Run proportionate checks.
10. Commit for review, report the meaningful changes and stop.

Keep the change small and understandable. Do not turn a module cleanup into a
major engineering project, expand to other modules, invent new requirements or
preserve every theoretical edge case. For each check, ask whether the supported
experiment actually needs it; remove it when it does not.

Summarize removals, behavior changes, validation and remaining limitations in the
final response. Keep experimental evidence needed for reproducibility without
adding tracing or a large validation project for routine readability work.

## Documentation policy

Do not create a new documentation report for ordinary refactoring, cleanup,
bug fixes or test maintenance. Summarize changes and test results in the final
Codex response. Update existing documentation when current behavior, interfaces,
configuration or usage changes.

Create a dedicated research document only for a substantial new experiment or
algorithm investigation with results worth preserving. Extend an existing
research document when the topic is already covered. Focus on methodology,
measurements, conclusions, limitations and reproduction; omit implementation
journals, permissions, temporary branches, routine test totals and repetitive
Git status statements.

Branch setup and guideline revision authorize no production refactor, test deletion,
Pyright installation or Isaac launch. The user will select the first module in a
separate task. Preserve the established SDK, launchers, workstation compatibility,
cameras and resource budgets described in AGENTS.md.
