"""Closed-loop improvement: turn a failure into a typed, reviewable change.

The pipeline is deliberately conservative, and the conservatism is the design
rather than a safety veneer over something looser. Three commitments shape every
module here:

**A failure has exactly one owning layer.** The hardest part of an improvement
loop is not generating a patch, it is generating the *right* patch. A loop that
patches whatever is easiest to patch -- the prompt -- will make the score move
while leaving the cause in place, and the next scenario on the same cause fails
again. The pipeline therefore classifies the cause first and treats the patch
type as a consequence of the classification, never as an input.

**The loop cannot see the holdout.** A loop that can see every test it is scored
against eventually optimises the tests. The holdout set is excluded from the
pipeline's inputs by construction (see ``clinical_scheduler.eval.scenarios``),
and the acceptance gate is the only thing that reads it. A patch that improves
the development sets while regressing the holdout is the specific failure this
arrangement exists to make visible.

**No patch reaches production.** A candidate is a signed artifact that has been
validated, scored, and approved; accepting it produces a new policy version and
a rollback target. Approval is simulated in this prototype and recorded as
simulated -- the difference between "accepted into demo policy vX" and
"committed to production" is not cosmetic, and papering over it would make every
downstream claim about the loop untrustworthy.

Entry point: :func:`clinical_scheduler.improvement.optimizer.run_repair`.
"""

from clinical_scheduler.improvement.taxonomy import (
    FailureClass,
    FailureClassification,
    PatchLayer,
    classify,
)

__all__ = [
    "FailureClass",
    "FailureClassification",
    "PatchLayer",
    "classify",
]
