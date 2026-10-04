# specs/ - layout and ID conventions

This folder holds the project constitution and the dated feature packets written
against it. The short version: `R-` is a requirement, `T-` is a task, `V-` is a
check that proves a requirement, and the letters exist so a spec can be argued
about precisely -- "V-5 doesn't actually prove R-3" is a reviewable claim in a way
that "the drain test looks weak" is not.

The same conventions are used across the neuriplo ecosystem; the reference
copy lives in [neuriplo](https://github.com/olibartfast/neuriplo). Cross-repository
work is specified in
[neuriplo-platform](https://github.com/olibartfast/neuriplo-platform) under
`specs/`; this folder covers work inside `neuriplo-kserve-runtime` and links
to platform packets when a phase is part of one.

## What is in here

| File | Role |
| --- | --- |
| `mission.md` | What the runtime is, who it serves, what it must never do. |
| `tech-stack.md` | Language, build, dependencies, architectural boundaries, explicit non-choices, validation entrypoints. |
| `roadmap.md` | Ordered phases with status. Its **Status Key** section defines the status values -- read it there rather than here, so the two cannot drift. |
| `YYYY-MM-DD-feature-name/` | One feature packet per increment of active work. The date is when the spec was written, not when it shipped. None exist yet. |

`roadmap.md`'s **Specification Rule** decides when a packet is required at all:
multi-phase work, public-behavior or architecture changes, or low reversibility.
Trivial fixes need no packet; small contained ones may get a PR-level note.

`plan/` is not replaced by this folder. It stays in place as the historical
implementation record (`plan/STEP0.md` to `plan/STEP14.md`, `plan/ROADMAP.md`,
`plan/NEXT_STEPS.md`).

## Packet files

| File | Answers | Owns IDs |
| --- | --- | --- |
| `requirements.md` | What must be true, and what is deliberately excluded | `R-`, `D-`, `A-`, `Q-` |
| `plan.md` | In what order, in thin phases that each end runnable | `T-` |
| `validation.md` | How each requirement is proven -- **written before implementation** | `V-`, `M-` |
| `orchestration.md` | How the phase is delegated to coding agents, when it is | `O-` |

`orchestration.md` is optional and appears only where a phase is actually being
delegated: roles, handoff packets, permission boundaries, the single acceptance
command, and the run ledger.

## The prefixes

| Prefix | Means | Lives in | Example |
| --- | --- | --- | --- |
| `R-n` | **Requirement.** Something that must be true when the phase is done. In scope, testable, and traceable to a `V-n`. | `requirements.md` | `[R-3]` a retired scheduler is drained exactly once |
| `D-n` | **Decision.** A choice already made, with its rationale, so it is not silently relitigated later. | `requirements.md` | `[D-4]` no change to the KServe V2 response shape |
| `A-n` | **Assumption.** Believed true but unverified, with its basis and how it will be confirmed. If an `A-` turns out false, something downstream is wrong -- that is the point of naming it. | `requirements.md` | `[A-1]` the stub executor needs no backend SDK |
| `Q-n` | **Open question.** Not yet decided, with a recommendation and who owns the call. A `Q-` that blocks work says so explicitly. | `requirements.md` | `[Q-2]` reject the request or reconcile the shape? |
| `T-n` | **Task.** A concrete unit of implementation work, grouped into thin phases. | `plan.md` | `[T-10]` accept a dynamic dimension in the codec |
| `V-n` | **Validation check.** An automated, observable proof of one or more requirements -- a command, a test, an assertion. | `validation.md` | `[V-5]` `ctest --preset debug` passes the codec cases |
| `M-n` | **Manual check.** A check that needs human judgement and cannot be automated honestly, such as a GPU run on a local machine. | `validation.md` | `[M-1]` `scripts/e2e-multi-backend.sh` on a GPU host |
| `O-n` | **Orchestration question.** An open question about delegation, permissions, or harness configuration rather than about the feature. | `orchestration.md` | `[O-1]` which agent owns the CI-only checks? |

## How they connect

The traceability rule is that every requirement has at least one check, and
every check names what it proves:

```
[R-3] a dynamic dimension in model metadata accepts any concrete extent
  -> [T-10] apply the dynamic-axis rule in NeuriploExecutor
  -> [V-5] -> [R-3]: unit test sends extent 640 against dim -1; rejection fails
```

Written in the files, a check states its targets with an arrow:

```
- [ ] [V-6] -> [R-10], [A-2]: real-onnx YOLO smoke returns the expected shapes
```

So `V-6` proves requirement `R-10` and simultaneously tests assumption `A-2`.
A requirement with no `V-` is unprovable and a `V-` with no `R-` is
unmotivated; both are review findings.

## Rules that keep the IDs useful

- **IDs are stable and append-only.** Never renumber. `R-7` means the same thing
  for the life of the packet, because commits, PR comments, and review threads
  cite it. A new requirement gets the next free number even if that leaves the
  list in a non-obvious order.
- **Split with a letter suffix.** When one check turns out to be two, add
  `V-4a` beside `V-4` rather than renumbering everything after it. Same for
  tasks (`T-27a`).
- **Retired IDs stay retired.** If a requirement is dropped, say so where it was
  and do not reuse the number.
- **Numbers are per packet.** `R-1` in one feature directory is unrelated to
  `R-1` in another. Cross-packet references name the packet.
- **Phases vs groups.** `roadmap.md` numbers repository phases. Inside a packet,
  `plan.md` numbers its own delivery slices as `Group 0`, `Group 1`, ... Each
  group ends in something runnable, so `develop` stays green mid-phase.
- **Validation before implementation.** `validation.md` is written and reviewed
  before code exists. A check invented after the fact tends to describe what the
  code happens to do rather than what the requirement demanded.
- **Evidence is recorded, not assumed.** `validation.md` carries an evidence
  table filled in with real results and dates. A checked box with no evidence
  row is not validated.

## Where this comes from

The packet structure and the validation-before-code rule follow the
`apply-spec-driven-development` workflow; the delegation and permission material
in `orchestration.md` follows `orchestrate-ai-coding-workflows`. The conventions
above are what those produce in this repository -- they are recorded here so the
packets are readable without the skills at hand.

No packet directory exists in this repository as of 2026-10-04. The in-flight
cross-repo packet for the current phase lives in neuriplo-platform and is linked
from `roadmap.md`.
