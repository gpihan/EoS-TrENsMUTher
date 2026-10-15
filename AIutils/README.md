# AI and LLM resources

This directory contains project-specific instructions for AI assistants working
with EoS-TrENsMUTher. Enumerate this directory at the start of a repository
task, then read only the resources relevant to the request.

These files are framework-neutral guidance, not permission to run jobs, submit
Slurm work, overwrite results, or invent missing scientific information. The
user's request remains authoritative. For software behavior and supported
fields, prefer the current source code, `docs/`, and `templates/` over an older
or conflicting AI note, and report any conflict rather than guessing.

## Resource routing

| Task | Read |
| --- | --- |
| Create, repair, or audit EoS metadata | [`MDSkillsAI.md`](MDSkillsAI.md) |
| Configure or execute preprocessing/inversion | [`RunSkillsAI.md`](RunSkillsAI.md) |
| Read, filter, reshape, or plot outputs | [`OutputSkillsAI.md`](OutputSkillsAI.md) |
| Configure or interpret a first-order EoS | [`FirstOrderSkillsAI.md`](FirstOrderSkillsAI.md), and also `MDSkillsAI.md` when editing metadata |

For a task spanning several stages, read the corresponding resources in
workflow order: metadata, run, first-order details when applicable, then
output interpretation.

## Shared rules

- Never infer undocumented units, normalizations, phase conventions, critical
  points, or column meanings.
- Distinguish an exact inverse root from a projected nearest admissible state.
- Preserve status, proof, phase, and branch-selection provenance.
- Treat `inverse.eosinv` as the authoritative result; the generic text table is
  a convenient scientific view.
- Use bounded inspection first. Do not start an expensive production run or
  submit an external job unless the user requested it.
- Do not silently alter solver tolerances or scientific semantics to make a
  table or run pass.

Human-facing documentation remains under [`docs/`](../docs/), reusable input
files under [`templates/`](../templates/), and runnable bounded examples under
[`examples/`](../examples/).
