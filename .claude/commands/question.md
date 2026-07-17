Update the investigation tracker at docs/investigations/OPEN_QUESTIONS.md.

Arguments: `$ARGUMENTS` — first token is a question ID (e.g. `Q3`, `Q6`); the
rest is a free-text update describing new evidence, a status change, or a
resolution.

Steps:

1. Read docs/investigations/OPEN_QUESTIONS.md and locate the `### <ID>:` entry
   (it may be under `## Open` or `## Resolved`). If no entry matches the ID,
   STOP and report that the ID doesn't exist — do not invent one. (To create a
   new question, say so explicitly and follow the entry structure in the file.)

2. Interpret the update and apply the MINIMAL change that fits the convention:
   - **Add evidence:** append the artifact filename / log path / SHA to the
     entry's `**Evidence:**` line (keep prior evidence; comma- or
     newline-separate).
   - **Change status:** edit the `**Status:**` line to `open`, `investigating`,
     or `resolved` as the update dictates.
   - **Resolve:** set `**Status:** resolved`, fill `**Resolution:**` with the
     answer + the evidence that settled it + `<YYYY-MM-DD> (<short SHA>)`, then
     MOVE the entire entry from `## Open` to the `## Resolved` section. Do not
     delete it.

3. Hard rules (from the file's own convention — enforce them):
   - NEVER edit an existing non-empty `**Resolution:**`. If the update overturns
     a resolved answer, instead create a NEW question that supersedes it, set the
     old entry's note to cross-link the new ID, and add the new ID's entry
     cross-linking back. Add a `## Corrections to project memory` entry if a
     prior belief was overturned.
   - For dates/SHAs, use today's date and `git rev-parse --short HEAD`. Do not
     fabricate dates for fields marked `(carry prior date)` — leave them unless
     the update supplies the real date.

4. Show the diff (`git diff docs/investigations/OPEN_QUESTIONS.md`) and STOP.
   Do not commit unless the user asks. Do not run any eval or training.
