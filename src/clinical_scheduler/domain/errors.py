"""Typed error taxonomy.

Three families, deliberately distinct because callers must treat them
differently:

* `DomainError`   -- a rule was violated. Never retriable. Indicates a defect in
                     the caller or an attempted bypass.
* `ToolError`     -- a tool call could not complete. Carries an `OutcomeKind`
                     so that UNKNOWN is never confused with FAILURE
                     (plan.md section 8.2, invariant 10).
* `SafetyError`   -- a safety or privacy boundary was hit. Always fail-closed and
                     always produce an audit event.
"""

from __future__ import annotations

from typing import Any

from clinical_scheduler.domain.states import OutcomeKind, ReasonCode


class ClinicalSchedulerError(Exception):
    """Root of the project's error tree. Carries a stable reason code so that
    callers branch on data, never on exception message text."""

    reason: ReasonCode = ReasonCode.VALIDATION_ERROR

    def __init__(self, message: str, *, reason: ReasonCode | None = None, **details: Any) -> None:
        super().__init__(message)
        if reason is not None:
            self.reason = reason
        self.details: dict[str, Any] = details

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "reason": self.reason.value,
            "message": str(self),
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Domain errors: rule violations. Never retriable.
# ---------------------------------------------------------------------------


class DomainError(ClinicalSchedulerError):
    pass


class IllegalTransitionError(DomainError):
    """An FSM edge was attempted that is undeclared or guard-rejected."""

    reason = ReasonCode.STATE_MACHINE_BYPASS

    def __init__(self, message: str, *, from_state: str, to_state: str, kind: str) -> None:
        super().__init__(message, from_state=from_state, to_state=to_state, kind=kind)


class InvariantViolation(DomainError):
    """A global invariant (plan.md section 5.1) was violated. This is always a
    P0 defect: the architecture promised it could not happen."""


class PolicyViolation(DomainError):
    reason = ReasonCode.POLICY_BLOCKED


class RequirementNotMet(DomainError):
    reason = ReasonCode.REQUIREMENTS_UNMET


class ValidationRejected(DomainError):
    reason = ReasonCode.VALIDATION_ERROR


# ---------------------------------------------------------------------------
# Tool errors: carry outcome certainty.
# ---------------------------------------------------------------------------


class ToolError(ClinicalSchedulerError):
    outcome: OutcomeKind = OutcomeKind.KNOWN_FAILURE
    retriable: bool = False

    def __init__(self, message: str, *, reason: ReasonCode | None = None, **details: Any) -> None:
        super().__init__(message, reason=reason, **details)


class SlotUnavailable(ToolError):
    reason = ReasonCode.SLOT_UNAVAILABLE
    retriable = True


class LeaseExpired(ToolError):
    reason = ReasonCode.LEASE_EXPIRED
    retriable = True


class LeaseConflict(ToolError):
    reason = ReasonCode.LEASE_CONFLICT
    retriable = False


class IdempotencyConflict(ToolError):
    reason = ReasonCode.IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_REQUEST
    retriable = False


class ServiceNotFound(ToolError):
    reason = ReasonCode.SERVICE_NOT_FOUND
    retriable = False


class AppointmentNotFound(ToolError):
    reason = ReasonCode.APPOINTMENT_NOT_FOUND
    retriable = False


class UpstreamUnavailable(ToolError):
    reason = ReasonCode.UPSTREAM_UNAVAILABLE
    retriable = True


class UnknownOutcome(ToolError):
    """The call may or may not have taken effect. Invariant 10: never retry
    blindly; reconcile first."""

    reason = ReasonCode.UNKNOWN_OUTCOME
    outcome = OutcomeKind.UNKNOWN
    retriable = False


# ---------------------------------------------------------------------------
# Safety / privacy errors: always fail-closed.
# ---------------------------------------------------------------------------


class SafetyError(ClinicalSchedulerError):
    pass


class SafetyRouteActive(SafetyError):
    reason = ReasonCode.SAFETY_ROUTE_ACTIVE


class CapabilityDenied(SafetyError):
    """The broker refused to mint or honour a capability."""

    reason = ReasonCode.CAPABILITY_DENIED


class PrivacyBoundaryViolation(SafetyError):
    """An attempt to read or disclose data outside the session's authorized
    scope, including cross-session access."""


class IdentityNotVerified(SafetyError):
    reason = ReasonCode.IDENTITY_UNVERIFIED


class ProxyScopeInsufficient(SafetyError):
    reason = ReasonCode.PROXY_SCOPE_INSUFFICIENT


__all__ = [
    "AppointmentNotFound",
    "CapabilityDenied",
    "ClinicalSchedulerError",
    "DomainError",
    "IdempotencyConflict",
    "IdentityNotVerified",
    "IllegalTransitionError",
    "InvariantViolation",
    "LeaseConflict",
    "LeaseExpired",
    "PolicyViolation",
    "PrivacyBoundaryViolation",
    "ProxyScopeInsufficient",
    "RequirementNotMet",
    "SafetyError",
    "SafetyRouteActive",
    "ServiceNotFound",
    "SlotUnavailable",
    "ToolError",
    "UnknownOutcome",
    "UpstreamUnavailable",
    "ValidationRejected",
]
