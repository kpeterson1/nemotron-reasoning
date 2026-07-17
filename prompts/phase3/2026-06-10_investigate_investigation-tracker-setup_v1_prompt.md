CC nailed it. My prompt had a bad premise — I pulled the base-model numbers
(9.5% / 11.9%) as if they were trained-model numbers. The actual picture is
completely different: bit_manip: 88% solver coverage → 33% trained accuracy. The
solver isn't the bottleneck. The model has the traces and can't learn them. This
is exactly where min-logprob inspection pays off. eq_trans: 28.6% ceiling
confirmed, but the 49 uncovered problems are variable-length symbolic transforms
that Huikang doesn't solve either. Realistic solver gains are ~3-5 numeric bug
fixes. CC's option 1+3 is the right call: Tell CC:

Go with option 1 (fix tractable eq_trans numeric bugs) and option 3 (investigate
bit_manip 88%→33% learning gap).

For option 3 specifically: run the v9 training traces for bit_manipulation
through the tokenizer and compute per-token log-likelihood against the trained
v9 adapter. Identify: (a) which of the 74 covered problems the model gets WRONG
at inference despite having correct training traces, (b) within those problems,
which tokens have the worst logprob. Report the problem IDs, the failing token
positions, and what those tokens represent in the trace structure. This is the
min-logprob inspection from Huikang's workflow — we need to see WHERE in the
trace the model chokes before we can redesign the format.

Phase discipline: do the eq_trans bug fixes first (small, contained,
single-variable). Then the bit_manip logprob investigation (read-only
diagnostic, no training). Stop and report both before any dataset rebuild.
