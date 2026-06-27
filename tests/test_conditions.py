"""Unit tests for conditions.py — the control-flow predicates."""

from conditions import (
    eval_condition,
    is_stop_signal,
    parse_equality,
    parse_when,
    token_present,
    when_skips,
)


def test_parse_equality_splits_field_and_value():
    assert parse_equality("decision == REJECTED") == ("decision", "REJECTED")


def test_parse_equality_malformed_returns_none():
    assert parse_equality("no operator here") is None


def test_parse_when_splits_pattern_and_id():
    assert parse_when("APPROVED in reviewer") == ("APPROVED", "reviewer")


def test_parse_when_malformed_returns_none():
    assert parse_when("garbage") is None


def test_eval_condition_match_and_miss():
    assert eval_condition("x == 1", {"x": "1"}) is True
    assert eval_condition("x == 1", {"x": "2"}) is False
    assert eval_condition("x == 1", {}) is False


def test_token_present_whole_word_only():
    assert token_present("APPROVED", "Looks good. APPROVED") is True
    assert token_present("APPROVED", "Still UNAPPROVED") is False


def test_is_stop_signal_only_on_final_line():
    assert is_stop_signal("done\nSTOP") is True
    assert is_stop_signal("STOP.") is True
    assert is_stop_signal("I did not STOP the process") is False
    assert is_stop_signal("NONSTOP") is False
    assert is_stop_signal("") is False


def test_when_skips_true_when_pattern_absent():
    assert when_skips("FOUND in card", {"card": "nothing here"}) is True


def test_when_skips_false_when_pattern_present():
    assert when_skips("FOUND in card", {"card": "FOUND it"}) is False


def test_when_skips_true_when_ref_missing():
    assert when_skips("FOUND in card", {}) is True
