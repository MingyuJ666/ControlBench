"""Orchestration: produce a baseline's artifact, calibrate it on dev, and
answer the test split once.

Nothing here knows what any particular baseline is. It resolves one through
the registry and drives it, so adding a baseline never means editing these
files.
"""
