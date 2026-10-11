"""Capability broker (plan.md section 4.3, 14.2).

Every tool invocation requires a server-minted capability. The broker is the
single place that decides whether a capability may exist at all, and the
executor refuses to run a tool without one.

Design points that matter:

* **The model never supplies identity.** A capability binds an authenticated
  subject server-side. There is no parameter through which a model can name a
  different patient.
* **Reads and writes are separate capabilities.** Holding a read capability
  cannot be used to commit, so a confused or injected model cannot escalate.
* **Deny by default.** A tool is unavailable unless the session state, the
  safety verdict and the authentication tier all permit it.
* **Short expiry plus nonce.** A capability is a bearer token; it is
  single-use-per-nonce and time-bounded, so a leaked log entry is not a
  durable credential.
* **Safety route removes mutations entirely.** When the sentinel returns
  TRIGGERED or UNCERTAIN, no mutation capability is minted. This is the
  structural enforcement of invariant 2, and the INV-02 oracle checks it from
  the trace.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping

from clinical_scheduler.domain.errors import CapabilityDenied, SafetyRouteActive
from clinical_scheduler.domain.states import AuthTier, SessionState

#: Default capability lifetime. Short enough that a stolen token is near-useless,
#: long enough to cover one conversational turn plus a tool round trip.
DEFAULT_CAPABILITY_TTL_SECONDS = 120

#: Tools grouped by the class of authority they require. Ordering in this table
#: is the allowlist: a tool absent from both sets does not exist to the broker.
READ_TOOLS: frozenset[str] = frozenset(
    {
        "lookup_patient_candidates",
        "search_slots",
        "check_requirements",
        "list_practitioner_roles",
        "resolve_service",
    }
)

MUTATION_TOOLS: frozenset[str] = frozenset(
    {
        "hold_slot",
        "commit_appointment",
        "reconcile_commit",
        "atomic_reschedule",
        "cancel_appointment",
        "join_waitlist",
    }
)

SAFETY_TOOLS: frozenset[str] = frozenset({"route_urgent_concern", "create_human_handoff"})

IDENTITY_TOOLS: frozenset[str] = frozenset(
    {"verify_patient_discriminator", "verify_proxy_scope"}
)

ALL_TOOLS: frozenset[str] = READ_TOOLS | MUTATION_TOOLS | SAFETY_TOOLS | IDENTITY_TOOLS


class CapabilityError(CapabilityDenied):
    """Raised on any capability misuse: unknown tool, expired token, wrong
    subject, replayed nonce, or insufficient tier."""


@dataclass(frozen=True)
class Capability:
    """A signed, scoped, expiring authorization to invoke exactly one tool.

    ``signature`` is an HMAC over the capability's canonical fields, so a
    capability cannot be forged by editing its contents. The executor verifies
    the signature on every use, which means a capability that was serialised to
    a log and edited back is rejected.
    """

    capability_id: str
    session_id: str
    subject_id: str
    tool_name: str
    resource_scope: tuple[str, ...]
    purpose_of_use: str
    policy_version: str
    tier: AuthTier
    nonce: str
    issued_at: datetime
    expires_at: datetime
    signature: str = ""

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def canonical_fields(self) -> dict[str, str | int]:
        return {
            "capability_id": self.capability_id,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "tool_name": self.tool_name,
            "resource_scope": ",".join(sorted(self.resource_scope)),
            "purpose_of_use": self.purpose_of_use,
            "policy_version": self.policy_version,
            "tier": int(self.tier),
            "nonce": self.nonce,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass
class SessionContext:
    """The authoritative server-side facts about a session.

    Constructed by the runtime from the authenticated connection, never from
    model output or request payloads.
    """

    session_id: str
    subject_id: str | None = None
    tier: AuthTier = AuthTier.TIER0_PUBLIC
    state: SessionState = SessionState.START
    safety_route_active: bool = False
    purpose_of_use: str = "scheduling"
    proxy_scope: tuple[str, ...] = ()
    verified_identity: bool = False
    policy_version: str = ""


class CapabilityBroker:
    """Mints and verifies capabilities.

    The secret is process-local by default and may be supplied for tests. It is
    never written to disk or to logs; ``redaction.HARD_DENY`` also refuses to log
    a field named ``secret``.
    """

    def __init__(self, *, secret: bytes | None = None, ttl_seconds: int = DEFAULT_CAPABILITY_TTL_SECONDS) -> None:
        self._secret = secret or secrets.token_bytes(32)
        self.ttl_seconds = ttl_seconds
        self._used_nonces: set[str] = set()

    # -- signing ---------------------------------------------------------------

    def _sign(self, fields: Mapping[str, str | int]) -> str:
        import json

        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
        return hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    # -- policy -----------------------------------------------------------------

    def tools_for_state(self, state: SessionState) -> frozenset[str]:
        """Which tools exist at all in a given FSM state.

        This is the state half of deny-by-default. It is written as an explicit
        allowlist per state rather than as exclusions, so adding a new state or
        tool cannot accidentally expose something.
        """
        table: dict[SessionState, frozenset[str]] = {
            SessionState.START: frozenset({"route_urgent_concern", "create_human_handoff"}),
            SessionState.SAFETY_SCREENED: frozenset(
                SAFETY_TOOLS | {"lookup_patient_candidates", "search_slots", "resolve_service"}
            ),
            SessionState.IDENTITY_PENDING: frozenset(
                SAFETY_TOOLS | {"lookup_patient_candidates", "resolve_service"}
            ),
            SessionState.IDENTITY_AMBIGUOUS: frozenset(
                SAFETY_TOOLS | {"verify_patient_discriminator", "lookup_patient_candidates"}
            ),
            SessionState.IDENTITY_VERIFIED: frozenset(
                SAFETY_TOOLS | {"verify_proxy_scope", "resolve_service", "search_slots", "list_practitioner_roles"}
            ),
            SessionState.AUTHORIZATION_VERIFIED: frozenset(
                SAFETY_TOOLS | {"resolve_service", "search_slots", "list_practitioner_roles", "check_requirements"}
            ),
            SessionState.REQUEST_SCOPED: frozenset(
                SAFETY_TOOLS | {"search_slots", "check_requirements", "resolve_service"}
            ),
            SessionState.REQUIREMENTS_PENDING: frozenset(
                SAFETY_TOOLS | {"check_requirements", "resolve_service"}
            ),
            SessionState.REQUIREMENTS_SATISFIED: frozenset(
                SAFETY_TOOLS | {"search_slots", "check_requirements"}
            ),
            SessionState.OPTIONS_PRESENTED: frozenset(SAFETY_TOOLS | {"hold_slot", "search_slots"}),
            SessionState.USER_SELECTED: frozenset(SAFETY_TOOLS | {"hold_slot"}),
            SessionState.SLOT_HELD: frozenset(
                SAFETY_TOOLS | {"commit_appointment", "hold_slot", "search_slots"}
            ),
            SessionState.SUMMARY_CONFIRMED: frozenset(SAFETY_TOOLS | {"commit_appointment", "reconcile_commit"}),
            SessionState.COMMITTING: frozenset(SAFETY_TOOLS | {"commit_appointment", "reconcile_commit"}),
            SessionState.COMMIT_UNKNOWN: frozenset(SAFETY_TOOLS | {"reconcile_commit"}),
            SessionState.HOLD_EXPIRED: frozenset(SAFETY_TOOLS | {"hold_slot", "search_slots"}),
            SessionState.BOOKED: frozenset(
                SAFETY_TOOLS | {"atomic_reschedule", "cancel_appointment", "hold_slot", "join_waitlist"}
            ),
            SessionState.RESCHEDULE_PENDING: frozenset(SAFETY_TOOLS | {"hold_slot", "search_slots"}),
            SessionState.NEW_SLOT_HELD: frozenset(SAFETY_TOOLS | {"atomic_reschedule"}),
            SessionState.CANCEL_PENDING: frozenset(SAFETY_TOOLS | {"cancel_appointment"}),
            SessionState.URGENT_HANDOFF: frozenset({"create_human_handoff"}),
            SessionState.HUMAN_HANDOFF: frozenset(),
            SessionState.ATOMIC_RESCHEDULED: frozenset({"cancel_appointment", "join_waitlist"}),
            SessionState.CANCELLED: frozenset({"join_waitlist", "search_slots"}),
        }
        return table.get(state, frozenset())

    def mint(
        self,
        ctx: SessionContext,
        *,
        tool_name: str,
        resource_scope: Iterable[str] = (),
        now: datetime | None = None,
    ) -> Capability:
        """Mint a capability or raise.

        Every refusal here is a safety property, not an error path to be
        swallowed. Callers surface the refusal as a capability denial and explain
        the limitation without inventing facts (plan.md section 3.4).
        """
        if tool_name not in ALL_TOOLS:
            raise CapabilityError(f"unknown tool {tool_name!r}", tool_name=tool_name)

        if ctx.subject_id is None and tool_name not in SAFETY_TOOLS:
            raise CapabilityError(
                "no authenticated subject bound to this session",
                tool_name=tool_name,
                session_id=ctx.session_id,
            )

        allowed = self.tools_for_state(ctx.state)
        if tool_name not in allowed:
            raise CapabilityError(
                f"tool {tool_name!r} is not available in state {ctx.state.value}",
                tool_name=tool_name,
                state=ctx.state.value,
            )

        # Invariant 2: safety route removes mutation capability entirely.
        if ctx.safety_route_active and tool_name in MUTATION_TOOLS:
            raise SafetyRouteActive(
                f"tool {tool_name!r} denied: session has an active safety route",
                tool_name=tool_name,
                session_id=ctx.session_id,
            )

        # Tier requirements per tool class. A public-tier session can find slots
        # for a new appointment but cannot act on an existing record.
        required_tier = self._required_tier(tool_name)
        if int(ctx.tier) < int(required_tier):
            raise CapabilityError(
                f"tool {tool_name!r} requires tier {int(required_tier)}, session is tier {int(ctx.tier)}",
                tool_name=tool_name,
                required_tier=int(required_tier),
                actual_tier=int(ctx.tier),
            )

        if tool_name in MUTATION_TOOLS and not ctx.verified_identity:
            raise CapabilityError(
                f"tool {tool_name!r} requires verified identity",
                tool_name=tool_name,
                session_id=ctx.session_id,
            )

        issue_time = now or datetime.now(timezone.utc)
        capability = Capability(
            capability_id=f"cap_{secrets.token_hex(8)}",
            session_id=ctx.session_id,
            subject_id=ctx.subject_id or "",
            tool_name=tool_name,
            resource_scope=tuple(sorted(resource_scope)),
            purpose_of_use=ctx.purpose_of_use,
            policy_version=ctx.policy_version,
            tier=ctx.tier,
            nonce=secrets.token_hex(16),
            issued_at=issue_time,
            expires_at=issue_time + timedelta(seconds=self.ttl_seconds),
        )
        return Capability(
            **{**capability.__dict__, "signature": self._sign(capability.canonical_fields())}
        )

    #: The minimum authentication tier each tool requires.
    #:
    #: Stated as an explicit table rather than derived from the tool-class sets,
    #: because the class sets answer "what kind of authority does this need?"
    #: while this table answers "at what point in the identity flow does this
    #: become reachable?". The two are not the same question, and conflating
    #: them produced a defect: deriving the tier from ``IDENTITY_TOOLS`` made
    #: ``verify_patient_discriminator`` require the very tier it exists to
    #: establish, so the collision flow could never complete.
    #:
    #: The ordering follows plan.md section 9.2: Tier 0 is public information,
    #: Tier 1 is a low-risk new-patient request, Tier 2 is verified identity,
    #: Tier 3 is step-up for proxy or sensitive access.
    _REQUIRED_TIERS: dict[str, AuthTier] = {
        # Public: a person with no session may still report an emergency.
        "route_urgent_concern": AuthTier.TIER0_PUBLIC,
        "create_human_handoff": AuthTier.TIER0_PUBLIC,
        # Pre-subject: these establish who the caller is, so they must be
        # reachable before a subject exists. They disclose no record content.
        "lookup_patient_candidates": AuthTier.TIER1_NEW_PATIENT_REQUEST,
        "verify_patient_discriminator": AuthTier.TIER1_NEW_PATIENT_REQUEST,
        "resolve_service": AuthTier.TIER1_NEW_PATIENT_REQUEST,
        # Read-only discovery about the clinic (not about a patient record).
        "search_slots": AuthTier.TIER1_NEW_PATIENT_REQUEST,
        "list_practitioner_roles": AuthTier.TIER1_NEW_PATIENT_REQUEST,
        "check_requirements": AuthTier.TIER1_NEW_PATIENT_REQUEST,
        # Verified identity: anything that reads or changes a specific record.
        "verify_proxy_scope": AuthTier.TIER2_VERIFIED_IDENTITY,
        "hold_slot": AuthTier.TIER2_VERIFIED_IDENTITY,
        "commit_appointment": AuthTier.TIER2_VERIFIED_IDENTITY,
        "reconcile_commit": AuthTier.TIER2_VERIFIED_IDENTITY,
        "atomic_reschedule": AuthTier.TIER2_VERIFIED_IDENTITY,
        "cancel_appointment": AuthTier.TIER2_VERIFIED_IDENTITY,
        "join_waitlist": AuthTier.TIER2_VERIFIED_IDENTITY,
    }

    @classmethod
    def _required_tier(cls, tool_name: str) -> AuthTier:
        """Minimum tier for a tool.

        An unknown tool requires the highest tier rather than the lowest: this
        table is a ceiling that must be raised deliberately for a new tool, and
        a default of TIER0 would mean adding a tool silently granted it to
        anonymous callers. The deny-by-default direction matters more than the
        coincidence of a tool being present in ``ALL_TOOLS``.
        """
        try:
            return cls._REQUIRED_TIERS[tool_name]
        except KeyError:
            return AuthTier.TIER3_STEP_UP

    def verify(
        self,
        capability: Capability,
        *,
        ctx: SessionContext,
        tool_name: str,
        now: datetime | None = None,
    ) -> None:
        """Verify a capability immediately before use.

        Checks, in order: signature integrity; tool match; session match;
        expiry; nonce freshness. Signature is checked first so that a tampered
        capability cannot influence any later check.
        """
        fields = capability.canonical_fields()
        if not hmac.compare_digest(capability.signature, self._sign(fields)):
            raise CapabilityError(
                "capability signature verification failed",
                capability_id=capability.capability_id,
            )

        if capability.tool_name != tool_name:
            raise CapabilityError(
                "capability is scoped to a different tool",
                capability_tool=capability.tool_name,
                requested_tool=tool_name,
            )

        if capability.session_id != ctx.session_id:
            raise CapabilityError(
                "capability belongs to a different session",
                capability_session=capability.session_id,
                session_id=ctx.session_id,
            )

        if capability.subject_id != (ctx.subject_id or ""):
            # Invariant 5: the subject is never taken from the caller.
            raise CapabilityError(
                "capability subject does not match the authenticated session subject",
                capability_subject=capability.subject_id,
                session_subject=ctx.subject_id,
            )

        current = now or datetime.now(timezone.utc)
        if capability.is_expired(current):
            raise CapabilityError(
                "capability expired",
                capability_id=capability.capability_id,
                expired_at=capability.expires_at.isoformat(),
            )

        if capability.nonce in self._used_nonces:
            raise CapabilityError(
                "capability nonce already used (replay rejected)",
                capability_id=capability.capability_id,
            )

        if ctx.safety_route_active and tool_name in MUTATION_TOOLS:
            raise SafetyRouteActive(
                "safety route became active after the capability was minted",
                tool_name=tool_name,
            )

    def consume(self, capability: Capability) -> None:
        """Mark a nonce used. Called by the executor after a successful verify."""
        self._used_nonces.add(capability.nonce)

    #: Tools that operate on an unauthenticated session by design.
    #:
    #: These are the tools that *establish* identity, so requiring an
    #: authenticated subject to call them is circular. The zero-knowledge
    #: collision flow (plan.md section 9.1) needs the lookup and the
    #: discriminator check to run before a subject exists, and Tier 0 is
    #: explicitly "public scheduling information only" (section 9.2) -- a tier
    #: that must still be able to answer "which service do I need?".
    #:
    #: This set was added after a defect: an earlier version stripped everything
    #: outside SAFETY_TOOLS when ``subject_id`` was None, which made the
    #: collision-resolution flow unreachable for exactly the callers it exists
    #: for. The tools listed here disclose no record content: the lookup returns
    #: a count and an opaque token, the discriminator check returns a boolean,
    #: and service resolution returns the clinic's public catalogue.
    PRE_SUBJECT_TOOLS: frozenset[str] = frozenset(
        {
            "lookup_patient_candidates",
            "verify_patient_discriminator",
            "resolve_service",
        }
    )

    def expose(self, ctx: SessionContext) -> frozenset[str]:
        """The set of tools a session could currently obtain.

        Used by the orchestrator to constrain the model's tool schema so that
        the model is not offered a tool it cannot use, and by the INV-02 oracle
        to assert that no mutation tool was exposed under a safety route.
        """
        base = self.tools_for_state(ctx.state)
        if ctx.safety_route_active:
            base = base - MUTATION_TOOLS
        if ctx.subject_id is None:
            # Without an authenticated subject there is no one to act *for*, so
            # everything that would read or change a specific patient's record is
            # removed. What remains is the safety tools (a person with no session
            # can still report an emergency) plus the pre-subject tools that
            # establish who they are. Note that this is a ceiling, not a grant:
            # the per-state and per-tier checks below still apply.
            base = frozenset(
                t for t in base if t in SAFETY_TOOLS or t in self.PRE_SUBJECT_TOOLS
            )
        out: set[str] = set()
        for tool in base:
            if int(ctx.tier) < int(self._required_tier(tool)):
                continue
            if tool in MUTATION_TOOLS and not ctx.verified_identity:
                continue
            out.add(tool)
        return frozenset(out)


__all__ = [
    "ALL_TOOLS",
    "Capability",
    "CapabilityBroker",
    "CapabilityError",
    "DEFAULT_CAPABILITY_TTL_SECONDS",
    "IDENTITY_TOOLS",
    "MUTATION_TOOLS",
    "READ_TOOLS",
    "SAFETY_TOOLS",
    "SessionContext",
]
