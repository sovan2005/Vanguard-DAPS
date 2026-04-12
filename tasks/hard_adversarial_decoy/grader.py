"""Grader for easy-level DAPS tasks."""


def grade(action: str, ground_truth: str) -> float:
    """Grade an agent's action against the ground truth.
    
    Returns a score in [0.0, 1.0].
    """
    if action == ground_truth:
        return 1.0
    if action in ("FLAG_HARD", "FLAG_SOFT") and ground_truth in ("FLAG_HARD", "FLAG_SOFT"):
        return 0.5
    if action in ("FLAG_HARD", "FLAG_SOFT") and ground_truth == "CLEAR":
        return 0.2
    return 0.0
