# Feature Requirements — <feature name>

> Copy this file to `specs/YYYY-MM-DD-<feature-name>/requirements.md` and delete
> the quoted guidance. Write this *before* `plan.md` and `validation.md`.

Roadmap phase: <link to the phase in ../roadmap.md>
Branch: `feature/<name>` (from `develop`, PR back to `develop`)

## Goal

What user- or system-visible outcome does this feature create? One paragraph.

## In Scope

- Observable behavior included in this branch. Name the endpoints, schedulers,
  backends, model kinds, and presets actually covered.

## Out of Scope

- Related work deliberately deferred, and where it goes instead (a roadmap
  phase, a sibling repo, an issue). "Helpful" expansion beyond this list is a
  review finding.

## Decisions

- Choice and its rationale. Record the ones where the obvious implementation
  would violate [`../tech-stack.md`](../tech-stack.md) or
  [`../mission.md`](../mission.md), where a sibling repo could plausibly have
  owned the change instead, or where a wire/metric/manifest contract moves.

## Constraints and Context

- Constitution rules that apply (ownership boundaries, explicit non-choices,
  stable error/metric/log surfaces).
- Existing patterns to reuse from [`../architecture.md`](../architecture.md)
  (Strategy/factory/adapter boundary, scheduler/executor split, control vs
  data plane).
- Cross-repo impact: does this need a `neuriplo` / `neuriplo-tasks` /
  `neuriplo-infer` / `neuriplo-kserve-client` change first?
- Contract surfaces touched: V2 HTTP/gRPC shapes, `/metrics` names,
  `deploy/` manifests, `config.pbtxt` handling.

## Open Questions

- Anything that could materially change the feature and has not been answered.
  Surface these before writing the plan, not after implementing.
