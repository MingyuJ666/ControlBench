"""Which instructions a task is contrasted against when selecting a unit.

Kept beside the baselines rather than inside one of them: two methods need
the same answer, and a discriminative score is only interpretable if both
computed it against the same set.
"""

from kernel.tasks import TASKS


def other_tasks(task: str) -> list[str]:
    """Every task but this one, in the benchmark's declared order.

    The order is fixed so the contrast set is reproducible, and the full set
    is used rather than a sample: a unit that is specific to this instruction
    has to be unresponsive to all four of the others, not to a chosen few.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {tuple(TASKS)}")
    return [name for name in TASKS if name != task]
