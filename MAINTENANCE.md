# Maintenance policy

This repository is a versioned research corpus, not a news feed. Version 1.0 is
the stable baseline. Add material only when it supports an active RoboLLM,
Signoff, robot-cells, or research-application question.

## Evidence states

Every research statement must be distinguishable as one of these states:

1. **Verified claim** — the primary paper was fetched and read. Quantitative
   claims name the arXiv identifier next to the number.
2. **Unverified reading map** — only metadata, title, abstract, repository, or
   transcript discovery was inspected. The section explicitly says “not
   verified” and does not present discovered numbers as evidence.
3. **Planning estimate** — a project estimate rather than a paper result. It is
   labelled as an estimate and must not be promoted into a sourced claim.

If a source cannot be fetched, keep it in state 2. Do not reconstruct benchmark
values from memory or search snippets.

## Adding a paper

1. Confirm the canonical identifier and title.
2. Add one unique record to `paper/references.bib`.
3. File it in the narrowest relevant note.
4. State whether the paper was read in full.
5. Attach the arXiv identifier to every quantitative claim.
6. Record limitations, dataset conditions, and evaluation caveats.
7. Run `python3 scripts/validate_repository.py`.

## Release rule

- Patch release: corrections, metadata repairs, and clearer caveats.
- Minor release: a verified thematic note or a material expansion supporting an
  active experiment.
- Major release: a new corpus structure or evidence contract.

Unread-paper accumulation alone is not a reason to release.
