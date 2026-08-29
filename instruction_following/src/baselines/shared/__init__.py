"""Machinery every method may use, owned by none of them.

A file belongs here when more than one method needs it and no method should
be able to change it for the others: the search-plan types the kernel
enforces, the activation capture every direction is learned from, and the
residual-stream hook the addition-based methods install.
"""
