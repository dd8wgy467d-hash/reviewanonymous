"""Tracking Shuffled Objects streams.

Each task module exposes the same interface, N being the number of people:
    alphabet(n)      vocabulary size
    n_states(n)      dense classes (machine states)
    n_answers(n)     extra answer classes read at the last token (0 if none)
    head_sizes(n)    the NFSM head family
    head_codes(n)    one [n_states] code per head: that head's state in each machine state
    generate(key, B, n, T) -> (x [B, T] int32, y [B, T, n_states] one-hot[, ans [B] int32])
"""

from nfsm.data.tso import parcel, question, swap

TASKS = {"swap": swap, "parcel": parcel, "question": question}
