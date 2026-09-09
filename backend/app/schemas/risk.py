"""Risk / decision-support schemas.

These outputs are a RESEARCH DECISION-SUPPORT SIGNAL. They are not a medical
diagnosis, have not been clinically validated, and must not be used for
patient care. The disclaimer travels with the payload so no consumer of the
API can accidentally strip it.
"""

from __future__ import annotations

from app.schemas.common import ApiModel, ReferralPriority, RiskLevel

DISCLAIMER = (
    "Research decision-support signal. Not a diagnostic result. "
    "Not clinically validated. Not for patient care."
)


class RiskFactor(ApiModel):
    """One transparent term in the risk score. No magic numbers."""

    id: str
    name: str
    #: The underlying measurement, in its own units.
    raw_value: float
    #: `raw_value` mapped into [0, 1] where 1 = maximally concerning.
    normalised: float
    weight: float
    contribution: float
    rationale: str


class RiskAssessment(ApiModel):
    #: Weighted sum of factor contributions, in [0, 1].
    risk_score: float
    risk_level: RiskLevel
    referral_priority: ReferralPriority
    review_recommended: bool
    #: Human-readable reasons, generated from the rules that actually fired.
    reasons: list[str]
    factors: list[RiskFactor]
    #: The rule set used, echoed so results are reproducible.
    weights: dict[str, float]
    thresholds: dict[str, float]
    triggered_overrides: list[str] = []
    disclaimer: str = DISCLAIMER
