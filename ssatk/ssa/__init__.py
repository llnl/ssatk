"""Space-situational-awareness screening and conjunction utilities."""

from .conjunction import (
    CatalogConjunctionEvent,
    ClosestApproach,
    ConjunctionCandidate,
    catalog_conjunction_screen,
    coarse_conjunction_screen,
    encounter_frame,
    probability_of_collision,
    refine_closest_approach,
    relative_encounter_covariance,
)
from .trajectories import StateBatch, ArrayStateProvider, OrbitStateProvider
from .observer import ObserverStates, from_ground_station, observer_metrics, earth_occulted
from .pair_geometry import pair_metrics
from .pair_screening import PairScreenResult, screen_pairs
from .event_windows import EventWindow, EventWindowAccumulator, buffered_event_windows
from .workflows import (
    PairAnalysisConfig, PairAnalysisResult, EventMetricBatch,
    analyze_pairs, iter_event_metrics, estimate_event_rows,
)

__all__ = [
    "CatalogConjunctionEvent",
    "ClosestApproach",
    "ConjunctionCandidate",
    "catalog_conjunction_screen",
    "coarse_conjunction_screen",
    "encounter_frame",
    "probability_of_collision",
    "refine_closest_approach",
    "relative_encounter_covariance",
    "StateBatch", "ArrayStateProvider", "OrbitStateProvider",
    "ObserverStates", "from_ground_station", "observer_metrics", "earth_occulted",
    "pair_metrics", "PairScreenResult", "screen_pairs",
    "EventWindow", "EventWindowAccumulator", "buffered_event_windows",
    "PairAnalysisConfig", "PairAnalysisResult", "EventMetricBatch",
    "analyze_pairs", "iter_event_metrics", "estimate_event_rows",
]
