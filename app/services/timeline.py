from datetime import datetime


def earliest_supported_observation(
    observations,
    analyzer,
    min_confidence=0.60
):
    """
    Compare chronological satellite observations and identify
    the earliest observation where detected change confidence
    reaches the configured threshold.

    Supports Pydantic Observation objects as well as dictionaries.
    """

    def get_value(obj, key):
        # Pydantic model / normal object
        if hasattr(obj, key):
            return getattr(obj, key)

        # Dictionary fallback
        if isinstance(obj, dict):
            return obj.get(key)

        raise TypeError(
            f"Unsupported observation type: {type(obj).__name__}"
        )

    # Sort observations chronologically
    observations = sorted(
        observations,
        key=lambda x: get_value(x, "date")
    )

    comparisons = []

    # Need at least two observations for temporal comparison
    if len(observations) < 2:
        return {
            "status": "insufficient_observations",
            "earliest_supported_date": None,
            "confidence": 0.0,
            "comparisons": []
        }

    earliest_date = None
    earliest_confidence = 0.0

    for i in range(1, len(observations)):

        before = observations[i - 1]
        after = observations[i]

        before_date = get_value(before, "date")
        after_date = get_value(after, "date")

        before_path = get_value(before, "path")
        after_path = get_value(after, "path")

        # Run temporal change analysis
        result = analyzer.analyze(
            before_path,
            after_path
        )

        confidence = float(
            result.get("confidence", 0.0)
        )

        comparisons.append({
            "from": before_date,
            "to": after_date,
            "result": result
        })

        # First observation whose confidence crosses threshold
        if (
            earliest_date is None
            and confidence >= min_confidence
        ):
            earliest_date = after_date
            earliest_confidence = confidence

    return {
        "status": "ok",
        "earliest_supported_date": earliest_date,
        "confidence": earliest_confidence,
        "comparisons": comparisons
    }