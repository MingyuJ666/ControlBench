"""Selecting the unit that carries *this* instruction, not any instruction.

The extraction contrast is a prompt with one instruction against the same
prompt without it, and that is the right contrast for steering: at evaluation
time the prompt carries no instruction, so the vector has to transport the
model from where it is to where the instruction would have put it.

It is the wrong contrast for *choosing* a single sparse unit. The difference
between the two conditions is one added sentence, which is a large and
obvious change, so almost any unit that represents the prompt at all can
separate them. A method that must pick one feature out of 65,536, or a few
heads out of 1152, therefore picks whatever responds most to "an instruction
is present" — which is shared by all five tasks and specific to none.

Measured on this benchmark: the SAE's mean-difference ranking chose the same
feature at layer 33 for four of the five tasks, and ITI's probe accuracy
saturated at 1.000 on 279 of 1152 heads, so its top-K was decided by index
order inside the tie.

The fix separates two questions that had been answered by one number:

    which unit is about this instruction?   -> contrast against the OTHER
                                               instructions, so the shared
                                               "an instruction is present"
                                               component cancels
    what should be added there?             -> unchanged: the shift from no
                                               instruction to this one

Both use activations already captured; the other tasks' positives are the
same prompts with a different sentence appended.
"""

import torch


def other_task_mean(others: list[torch.Tensor]) -> torch.Tensor:
    """Mean activation across the tasks whose instruction is not the target.

    Averaging over tasks rather than pooling their rows keeps every task
    weighted equally, which matters because a unit that responds to one other
    instruction very strongly would otherwise dominate the comparison.
    """
    if not others:
        raise ValueError("a discriminative score needs at least one other task")
    return torch.stack([activations.mean(dim=0) for activations in others]).mean(dim=0)


def discriminative_separation(target: torch.Tensor,
                              others: list[torch.Tensor]) -> torch.Tensor:
    """How much more each unit responds to this instruction than to others.

    A unit that fires for any instruction scores about zero here, whatever it
    scored against the no-instruction condition.
    """
    return target.mean(dim=0) - other_task_mean(others)


def standardized_gap(target: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """The same difference in units of the pooled within-group spread.

    Accuracy saturates once two groups are separable at all — 279 heads
    reached 1.000 and could not be ordered — and so does AUC, which is the
    same question asked of the ranking. An effect size does not: two units
    that both separate perfectly still differ by how far apart they put the
    groups, which is what decides how much room a dose has.
    """
    spread = torch.sqrt(
        0.5 * (target.var(dim=0, unbiased=True)
               + others.var(dim=0, unbiased=True))
    )
    return (target.mean(dim=0) - others.mean(dim=0)) / spread.clamp_min(1e-6)
