"""Grader for easy-level DAPS tasks."""


def grade(action: str, ground_truth: str) -> float:
    """Grade an agent's action against the ground truth.
    
    Returns a score in [0.01, 0.99].
    """
    if action == ground_truth:
        return 0.99
    if action in ("FLAG_HARD", "FLAG_SOFT") and ground_truth in ("FLAG_HARD", "FLAG_SOFT"):
        return 0.50
    if action in ("FLAG_HARD", "FLAG_SOFT") and ground_truth == "CLEAR":
        return 0.20
    return 0.01
