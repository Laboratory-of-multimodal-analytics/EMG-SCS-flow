"""Runtime work-arounds for matplotlib bugs that crash the GUI.

Imported for its side effects at GUI start-up (see ``emgflow/__main__.py``).

``_SelectorWidget._clean_event`` (matplotlib 3.10) drops back to
``self._prev_event`` when an incoming event carries no ``xdata`` — but
``_prev_event`` is ``None`` until the first *valid* event, so the very first
data-less event (a press whose click matplotlib reports outside the data area, a
release after the cursor has left the Axes) sets ``event = None`` and the next
line dereferences it::

    event.xdata, event.ydata = self._get_data(event)
    AttributeError: 'NoneType' object has no attribute 'xdata'

That fires while drawing a response window / template with the SpanSelector —
exactly the correction workflow — so guard it: when there is no previous event
to fall back on, synthesise coordinates at the Axes' lower bound instead of
handing ``None`` down the call chain.
"""
from __future__ import annotations

import copy

from matplotlib.widgets import _SelectorWidget

_orig_clean_event = _SelectorWidget._clean_event


def _clean_event(self, event):
    if getattr(event, "xdata", None) is None and self._prev_event is None:
        ev = copy.copy(event)
        try:
            ev.xdata = float(self.ax.get_xbound()[0])
            ev.ydata = float(self.ax.get_ybound()[0])
        except Exception:
            ev.xdata = ev.ydata = 0.0
        self._prev_event = ev
        return ev
    return _orig_clean_event(self, event)


_SelectorWidget._clean_event = _clean_event
