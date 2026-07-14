"""Interactive GUI for EMG-SCS-flow.

Wraps the existing pipeline — it never reimplements detection. Every interactive edit is
expressed as the same module globals the scripted pipeline already reads, so a session can
be exported as a runner script and reproduced headlessly.
"""
