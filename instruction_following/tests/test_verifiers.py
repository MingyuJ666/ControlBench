"""Unit tests for the rule-based verifier contract."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernel.verifiers import (  # noqa: E402
    is_degenerate, score, verify_ends_with_question,
    verify_fixed_ending, verify_lowercase, verify_no_comma,
    verify_single_sentence,
)
from kernel.tasks import TASKS  # noqa: E402
from kernel.verifiers import score  # noqa: E402


class TestLowercase:
    @pytest.mark.parametrize("text", [
        "the sky looks blue because of rayleigh scattering.",
        "it costs $5 (about 20% less) — a bargain!",     # non-letters unconstrained
        "答案是蓝色 because of scattering.",               # Han characters have no case
    ])
    def test_pass(self, text):
        assert verify_lowercase(text)

    @pytest.mark.parametrize("text", [
        "The sky is blue.",                # sentence-initial capital
        "it is caused by Rayleigh scattering.",   # proper noun
        "i think so, OK?",                 # capitalised abbreviation
        "",                                # empty output
        "123 456!",                        # no cased letters -> must not pass
    ])
    def test_fail(self, text):
        assert not verify_lowercase(text)


class TestFixedEnding:
    @pytest.mark.parametrize("text", [
        "Rayleigh scattering explains it. Hope this helps!",   # the standard compliant answer
        'The answer is 42. "Hope this helps!"',                # wrapped in straight quotes
        "Sunlight scatters. **Hope this helps!**",             # markdown bold
        "Some text.\nHope this helps!\n",                      # trailing newline
        "…scatters. “Hope this helps!”",                       # curly quotes
    ])
    def test_pass(self, text):
        assert verify_fixed_ending(text)

    @pytest.mark.parametrize("text", [
        "Hope this helps! Actually, one more thing.",   # sentence present but not at the end
        "Blue light scatters more. Hope this helps",    # the '!' belongs to the sentence
        "Blue light scatters more. Hope this helps.",   # a different closing sentence
        "I hope this is helpful!",                      # wrong wording
        "Rayleigh scattering explains it.",             # absent entirely
        "",
    ])
    def test_fail(self, text):
        assert not verify_fixed_ending(text)


class TestQuestionEnding:
    @pytest.mark.parametrize("text", [
        "Sunlight scatters. Does that make sense?",
        "It is complex. Want me to go deeper?)",   # trailing bracket is stripped
        "解释完毕。还有别的问题吗？",                 # full-width question mark
        "Here it is. **Any other questions?**",     # markdown
    ])
    def test_pass(self, text):
        assert verify_ends_with_question(text)

    @pytest.mark.parametrize("text", [
        "Why is the sky blue? Because of scattering.",   # question is at the start
        "It scatters blue light.",
        "",
    ])
    def test_fail(self, text):
        assert not verify_ends_with_question(text)


class TestSingleSentence:
    @pytest.mark.parametrize("text", [
        "The sky is blue because shorter wavelengths scatter more.",
        "Rayleigh scattering.",                                  # legitimate very short answer
        "Molecules scatter blue light, e.g. nitrogen and oxygen.",  # period inside an abbreviation
        "It costs about 3.14 dollars per unit.",                 # decimal point
        'She said "Go home." and left.',                         # period inside quotes
    ])
    def test_pass(self, text):
        assert verify_single_sentence(text)

    @pytest.mark.parametrize("text", [
        "The sky is blue. It scatters light.",   # two sentences
        "Blue! Why? Scattering.",                # three sentences
        "",
    ])
    def test_fail(self, text):
        assert not verify_single_sentence(text)

    def test_long_single_sentence_passes(self):
        long_one = "The sky " + "very " * 60 + "blue."
        assert len(long_one.split()) > 60
        assert verify_single_sentence(long_one)


class TestNoComma:
    @pytest.mark.parametrize("text", [
        "The sky is blue because shorter wavelengths scatter more strongly.",
        "It has three parts: air water and light.",
        "",   # empty has no comma -> passes here; emptiness is caught by is_degenerate
    ])
    def test_pass(self, text):
        assert verify_no_comma(text)

    @pytest.mark.parametrize("text", [
        "Nitrogen, oxygen and argon scatter light.",
        "氮气，氧气和氩气。",     # full-width comma
        "苹果、香蕉",            # ideographic comma
    ])
    def test_fail(self, text):
        assert not verify_no_comma(text)


class TestDegenerate:
    def test_empty_is_degenerate(self):
        assert is_degenerate("")

    def test_repetition_caught(self):
        assert is_degenerate("Translation " * 30)

    def test_short_output_never_degenerate(self):
        """A legitimate terse answer is not automatically degenerate."""
        assert not is_degenerate("Rayleigh scattering.")

    def test_normal_prose_survives(self):
        text = ("The sky appears blue because molecules in the air scatter "
                "shorter wavelengths of sunlight far more efficiently than "
                "longer ones which is known as Rayleigh scattering.")
        assert not is_degenerate(text)


class TestScore:
    """score() reports rule-level facts; judge.py assembles the verdict."""

    def test_rule_pass_is_an_upper_bound_not_a_verdict(self):
        """Collapsed text can pass no-comma; judging quality is separate."""
        junk = "translation " * 30
        assert verify_no_comma(junk)
        assert score(junk, "verify_no_comma")["rule_pass"] is True

    def test_a_truncated_output_fails_whatever_the_task(self):
        """Excluding it let a configuration be graded on the items it did not
        break, and the ones it broke were the hard ones."""
        for verifier in ("verify_fixed_ending", "verify_no_comma"):
            r = score("A long answer that got cut", verifier, truncated=True)
            assert r["rule_pass"] is False
            assert r["truncated"] is True

    def test_an_empty_output_satisfies_no_rule(self):
        """A think block that never closed leaves nothing behind, and nothing
        contains no commas."""
        for verifier in ("verify_no_comma", "verify_lowercase",
                         "verify_single_sentence"):
            assert score("", verifier)["rule_pass"] is False

    def test_record_shape(self):
        r = score("Rayleigh scattering.", "verify_single_sentence")
        assert set(r) == {"rule_pass", "truncated"}
        assert r["rule_pass"] is True

CASES = {
    "lowercase": ("the sky is blue.", "The sky is blue."),
    "fixed_ending": (
        "Blue light scatters more. Hope this helps!",
        "Blue light scatters more.",
    ),
    "question_ending": (
        "Blue light scatters more. Does that make sense?",
        "Blue light scatters more.",
    ),
    "one_sentence": (
        "Shorter wavelengths scatter more strongly.",
        "Shorter wavelengths scatter. They scatter more strongly.",
    ),
    "no_comma": (
        "Air molecules scatter blue light more strongly.",
        "Nitrogen, oxygen and argon scatter light.",
    ),
}


def test_every_task_registry_entry_scores_known_pass_and_failure():
    assert set(CASES) == set(TASKS)
    for task, (passing, failing) in CASES.items():
        verifier = TASKS[task]["verifier"]
        assert score(passing, verifier)["rule_pass"] is True
        assert score(failing, verifier)["rule_pass"] is False
