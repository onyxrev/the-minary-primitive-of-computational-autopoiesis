"""
Perspective definitions for the Minary simulation.

This module defines the n=5 perspectives and their competency profiles,
which together form the competency matrix C ∈ R^(n×m) from Paper Section 4.

Each perspective has competencies C_i,j ∈ [0, 1] for m=19 semantic dimensions.
These competencies are "hidden" from the system in that they only influence
behavior through the response calculation r(t)_i,j = x(t)_j - C_i,j.

Paper Reference: Section 7.1 (Setup)
    "Each perspective has been assigned a competency value in [0, 1] for
    each dimension; together, these roughly create profiles that reflect
    their respective 'archetypes' of the perspectives."

The archetypes are designed to create interesting competency asymmetries
that drive emergent behaviors like the halo effect (Section 7.4-7.5).
"""

from decimal import Decimal
from data_types import Perspective, SemanticProfile

# Placeholder perspective (not used in simulation)
TheSignal = Perspective(
    name="TheSignal",
    semantic_profile=SemanticProfile(semantics={}),
    semantic_adjustments={}
)

# -----------------------------------------------------------------------------
# THE FIVE PERSPECTIVE ARCHETYPES
# Each represents a different "expert" profile with distinct competency patterns
# -----------------------------------------------------------------------------

# Perspective 1: The True Artist (p_1)
# Creative visionary with high competency in artistic dimensions
# Strengths: illustration (0.95), modern art (0.95), character/costume design (0.8)
# Weaknesses: sentiment (0.1), brand voice (0.2), fashion trend (0.2)
TheTrueArtist = Perspective(
    name="The True Artist",
    semantic_profile=SemanticProfile(
        semantics={
            "3d modeling": Decimal("0.7"),
            "anatomy": Decimal("0.4"),
            "artwork similarity": Decimal("0.2"),
            "audience relevance": Decimal("0.6"),
            "brand voice": Decimal("0.2"),
            "character design": Decimal("0.8"),
            "color grading": Decimal("0.6"),
            "color scheme": Decimal("0.5"),
            "costume design": Decimal("0.8"),
            "fashion trend": Decimal("0.2"),
            "illustration": Decimal("0.9"),
            "information redaction": Decimal("0.2"),
            "interior design": Decimal("0.6"),
            "modern art": Decimal("0.95"),
            "photographic composition": Decimal("0.7"),
            "physics": Decimal("0.5"),
            "sentiment": Decimal("0.1"),
            "usability": Decimal("0.55"),
            "visual ad": Decimal("0.4")
        }
    ),
    semantic_adjustments={}
)

# Perspective 2: The Executive Director (p_2)
# Business-focused leader prioritizing brand and audience
# Strengths: brand voice (0.97), visual ad (0.9), audience relevance (0.8)
# Weaknesses: 3d modeling (0.1), anatomy (0.1), illustration (0.1)
TheExecutiveDirector = Perspective(
    name="The Executive Director",
    semantic_profile=SemanticProfile(
        semantics={
            "3d modeling": Decimal("0.1"),
            "anatomy": Decimal("0.1"),
            "artwork similarity": Decimal("0.7"),
            "audience relevance": Decimal("0.8"),
            "brand voice": Decimal("0.97"),
            "character design": Decimal("0.2"),
            "color grading": Decimal("0.1"),
            "color scheme": Decimal("0.7"),
            "costume design": Decimal("0.3"),
            "fashion trend": Decimal("0.7"),
            "illustration": Decimal("0.1"),
            "information redaction": Decimal("0.8"),
            "interior design": Decimal("0.5"),
            "modern art": Decimal("0.7"),
            "photographic composition": Decimal("0.5"),
            "physics": Decimal("0.3"),
            "sentiment": Decimal("0.8"),
            "usability": Decimal("0.7"),
            "visual ad": Decimal("0.9")
        }
    ),
    semantic_adjustments={}
)

# Perspective 3: The Technician (p_3)
# Technical expert focused on craft and precision
# Strengths: physics (0.95), color grading (0.95), 3d modeling (0.9)
# Weaknesses: audience relevance (0.1), sentiment (0.1), visual ad (0.3)
TheTechnician = Perspective(
    name="The Technician",
    semantic_profile=SemanticProfile(
        semantics={
            "3d modeling": Decimal("0.9"),
            "anatomy": Decimal("0.6"),
            "artwork similarity": Decimal("0.3"),
            "audience relevance": Decimal("0.1"),
            "brand voice": Decimal("0.3"),
            "character design": Decimal("0.7"),
            "color grading": Decimal("0.95"),
            "color scheme": Decimal("0.8"),
            "costume design": Decimal("0.3"),
            "fashion trend": Decimal("0.4"),
            "illustration": Decimal("0.6"),
            "information redaction": Decimal("0.7"),
            "interior design": Decimal("0.5"),
            "modern art": Decimal("0.5"),
            "photographic composition": Decimal("0.75"),
            "physics": Decimal("0.95"),
            "sentiment": Decimal("0.1"),
            "usability": Decimal("0.7"),
            "visual ad": Decimal("0.3")
        }
    ),
    semantic_adjustments={}
)

# Perspective 4: The Critic (p_4)
# Analytical evaluator focused on assessment and trends
# Strengths: sentiment (0.97), artwork similarity (0.9), fashion trend (0.9)
# Weaknesses: physics (0.1), illustration (0.2), color grading (0.2)
TheCritic = Perspective(
    name="The Critic",
    semantic_profile=SemanticProfile(
        semantics={
            "3d modeling": Decimal("0.3"),
            "anatomy": Decimal("0.6"),
            "artwork similarity": Decimal("0.9"),
            "audience relevance": Decimal("0.75"),
            "brand voice": Decimal("0.87"),
            "character design": Decimal("0.3"),
            "color grading": Decimal("0.2"),
            "color scheme": Decimal("0.7"),
            "costume design": Decimal("0.6"),
            "fashion trend": Decimal("0.9"),
            "illustration": Decimal("0.2"),
            "information redaction": Decimal("0.7"),
            "interior design": Decimal("0.5"),
            "modern art": Decimal("0.8"),
            "photographic composition": Decimal("0.5"),
            "physics": Decimal("0.1"),
            "sentiment": Decimal("0.97"),
            "usability": Decimal("0.7"),
            "visual ad": Decimal("0.83")
        }
    ),
    semantic_adjustments={}
)

# Perspective 5: The Fan (p_5)
# Audience proxy focused on user experience and appeal
# Strengths: audience relevance (0.95), sentiment (0.95), usability (0.85)
# Weaknesses: 3d modeling (0.1), visual ad (0.2), anatomy (0.2)
TheFan = Perspective(
    name="The Fan",
    semantic_profile=SemanticProfile(
        semantics={
            "3d modeling": Decimal("0.1"),
            "anatomy": Decimal("0.2"),
            "artwork similarity": Decimal("0.75"),
            "audience relevance": Decimal("0.95"),
            "brand voice": Decimal("0.7"),
            "character design": Decimal("0.5"),
            "color grading": Decimal("0.4"),
            "color scheme": Decimal("0.6"),
            "costume design": Decimal("0.5"),
            "fashion trend": Decimal("0.8"),
            "illustration": Decimal("0.3"),
            "information redaction": Decimal("0.3"),
            "interior design": Decimal("0.5"),
            "modern art": Decimal("0.6"),
            "photographic composition": Decimal("0.5"),
            "physics": Decimal("0.3"),
            "sentiment": Decimal("0.95"),
            "usability": Decimal("0.85"),
            "visual ad": Decimal("0.2")
        }
    ),
    semantic_adjustments={}
)

# The complete set of perspectives forming the competency matrix C ∈ R^(5×19)
# Each perspective is a row, each semantic dimension is a column
Perspectives = [TheTrueArtist, TheExecutiveDirector, TheTechnician, TheCritic, TheFan]
