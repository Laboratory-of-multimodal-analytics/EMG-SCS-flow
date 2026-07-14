"""Auto-generated settings form — one control per lever, grouped as in constants.py."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from ..settings_spec import Lever, levers_for


class _OptFloat(QLineEdit):
    """A float that can be empty (None) — the pipeline's 'None disables this gate'."""

    def __init__(self, value) -> None:
        super().__init__("" if value is None else str(value))
        self.setPlaceholderText("empty = off")

    def value(self):
        t = self.text().strip()
        return None if t == "" else float(t)


class SettingsPanel(QScrollArea):
    changed = Signal(str, object)  # lever name, new value

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self._widgets: dict[str, QWidget] = {}
        self.setWidgetResizable(True)
        self.rebuild()

    # ------------------------------------------------------------------ #
    def rebuild(self) -> None:
        self._widgets.clear()
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setAlignment(Qt.AlignTop)

        groups: dict[str, list[Lever]] = {}
        for lv in levers_for(self.session.mode):
            groups.setdefault(lv.group, []).append(lv)

        for group, levers in groups.items():
            box = QGroupBox(group)
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignRight)
            for lv in levers:
                w = self._make_widget(lv)
                self._widgets[lv.name] = w
                label = QLabel(lv.label)
                tip = lv.doc
                if lv.target == "default":
                    tip = (tip + "\n\n" if tip else "") + (
                        "Bound as a function default in pipeline.py and never passed by its "
                        "caller — the GUI rebinds the function's defaults to make this work."
                    )
                if tip:
                    label.setToolTip(tip)
                    w.setToolTip(tip)
                    label.setText(lv.label + " ⓘ")
                form.addRow(label, w)
            layout.addWidget(box)

        note = QLabel(
            "Leakage thresholds (±40 ms, corr 0.7, >3 channels) are hardcoded in pipeline.py "
            "and cannot be tuned here.\nSTARTSTOP_ONSET_TMIN / _ONSET_TMAX / _RESP_TMAX are "
            "imported but never read — they are deliberately not shown."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note)

        self.setWidget(body)

    # ------------------------------------------------------------------ #
    def _make_widget(self, lv: Lever) -> QWidget:
        value = self.session.get(lv.name)

        if lv.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v, n=lv.name: self._on_change(n, v))
            return w

        if lv.kind == "choice":
            w = QComboBox()
            w.addItems(list(lv.choices))
            if str(value) in lv.choices:
                w.setCurrentText(str(value))
            w.currentTextChanged.connect(lambda v, n=lv.name: self._on_change(n, v))
            return w

        if lv.kind == "int":
            w = QSpinBox()
            w.setRange(0, 1_000_000)
            w.setValue(int(value or 0))
            w.valueChanged.connect(lambda v, n=lv.name: self._on_change(n, v))
            return w

        if lv.kind == "float":
            w = QDoubleSpinBox()
            w.setRange(-1e9, 1e9)
            w.setDecimals(4)
            w.setSingleStep(0.01)
            w.setValue(float(value or 0.0))
            w.valueChanged.connect(lambda v, n=lv.name: self._on_change(n, v))
            return w

        if lv.kind == "opt_float":
            w = _OptFloat(value)
            w.editingFinished.connect(lambda n=lv.name, ww=None: self._on_change(n, self._widgets[n].text()))
            return w

        # "floats" (comma-separated tuple) and "str"
        text = ", ".join(str(v) for v in value) if isinstance(value, (tuple, list)) else str(value or "")
        w = QLineEdit(text)
        w.editingFinished.connect(lambda n=lv.name: self._on_change(n, self._widgets[n].text()))
        return w

    def _on_change(self, name: str, value) -> None:
        self.session.set(name, value)
        self.changed.emit(name, self.session.get(name))
