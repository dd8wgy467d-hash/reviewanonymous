"""Tracking Shuffled Objects streams. Each task module (N people) defines alphabet(n), n_states(n),
n_answers(n), head_sizes(n), head_codes(n) and generate(key, B, n, T).
"""

from nfsm.data.tso import parcel, question, swap

TASKS = {"swap": swap, "parcel": parcel, "question": question}
