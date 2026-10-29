"""Numeric economic transport shared by the runner, publisher and event hooks.

The fixed settings prefix ends at E_QTY_START. Variable quantity data is always
appended there; adding a setting must move that one boundary for every reader.
"""

E_POINT_VALUE = 0
E_FRICTION = 1
E_BPS = 2
E_MODE = 3
E_AMOUNT = 4
E_CAP = 5
E_LOT_STEP = 6
E_MIN_QTY = 7
E_CLAMP_MIN = 8
E_PROJECTED_OVERFLOW = 9
E_ENTRY_NOTIONAL = 10
E_CHECK_CASH = 11
E_ADMISSION = 12
E_QTY_START = 13
