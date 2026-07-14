"""Session state: the settings + manual edits that fully define one analysis run.

Design rule (from the toolbox spec): *every* interactive edit must be expressible as
the same globals the scripted pipeline already reads, so a settings file — or the
generated runner script — reconstructs the result exactly.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .settings_spec import BY_NAME, defaults_for

# Key precedence for the force dicts (most specific wins):
#   "channel"  <  (config, channel)  <  (config, amp, channel)
ForceKey = tuple  # ("channel",) | (config, channel) | (config, amp, channel)


def _coerce(name: str, value: Any) -> Any:
    """Turn a widget-level value into what the pipeline expects."""
    lever = BY_NAME.get(name)
    if lever is None:
        return value
    if lever.kind == "opt_float":
        if value in ("", None):
            return None
        return float(value)
    if lever.kind == "float":
        return float(value)
    if lever.kind == "int":
        return int(value)
    if lever.kind == "bool":
        return bool(value)
    if lever.kind == "floats":
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return tuple(float(p) for p in parts)
        return tuple(float(v) for v in value)
    if lever.kind == "str":
        return "" if value is None else str(value)
    return value


@dataclass
class Session:
    input_path: Path | None = None
    output_dir: Path | None = None
    mode: str = "sir"  # "sir" | "startstop"
    settings: dict[str, Any] = field(default_factory=dict)

    # Session-local template bank (set when the user makes a template from a selection).
    # None = the stock bank shipped with the repo.
    template_dir: Path | None = None
    template_only_user: bool = False

    # ---- manual edits (SIR) — the force/suppress/exclude system ----
    force_pos_p1: dict[ForceKey, Any] = field(default_factory=dict)
    force_neg_p1: dict[ForceKey, Any] = field(default_factory=dict)
    force_keys: set[tuple[str, str, str]] = field(default_factory=set)      # bypass the gates
    suppress_keys: set[tuple[str, str, str]] = field(default_factory=set)   # force-drop
    exclude_channels: set[str] = field(default_factory=set)
    exclude_configs: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.settings:
            self.settings = defaults_for(self.mode)

    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str) -> None:
        """Switch branch, keeping any settings the new mode shares."""
        if mode == self.mode:
            return
        self.mode = mode
        merged = defaults_for(mode)
        for k, v in self.settings.items():
            if k in merged:
                merged[k] = v
        self.settings = merged

    def get(self, name: str) -> Any:
        return self.settings.get(name, BY_NAME[name].default if name in BY_NAME else None)

    def set(self, name: str, value: Any) -> None:
        self.settings[name] = _coerce(name, value)

    # ------------------------------------------------------------------ #
    # Edits. Every one is ADDITIVE: it touches only the named
    # (config, amp, channel) and never overwrites unmentioned detections.
    # ------------------------------------------------------------------ #
    def force_p1(
        self,
        config: str,
        amp: str,
        channel: str,
        positive: bool,
        min_lat: float,
        max_lat: float | None = None,
    ) -> None:
        """Force P1 onto a deflection of the chosen polarity inside [min_lat, max_lat].

        A negative `min_lat` forces a peak BEFORE t=0 (used for early near-zero peaks).
        Writing the most specific 3-level key so the edit cannot leak onto other crops.
        """
        key = (config, amp, channel)
        value = min_lat if max_lat is None else (min_lat, max_lat)
        target = self.force_pos_p1 if positive else self.force_neg_p1
        other = self.force_neg_p1 if positive else self.force_pos_p1
        other.pop(key, None)  # a peak cannot be forced to both polarities
        target[key] = value

    def clear_force(self, config: str, amp: str, channel: str) -> None:
        key = (config, amp, channel)
        self.force_pos_p1.pop(key, None)
        self.force_neg_p1.pop(key, None)

    def toggle_suppress(self, config: str, amp: str, channel: str) -> bool:
        key = (config, amp, channel)
        if key in self.suppress_keys:
            self.suppress_keys.discard(key)
            return False
        self.suppress_keys.add(key)
        self.force_keys.discard(key)  # suppress and whitelist are mutually exclusive
        return True

    def toggle_whitelist(self, config: str, amp: str, channel: str) -> bool:
        key = (config, amp, channel)
        if key in self.force_keys:
            self.force_keys.discard(key)
            return False
        self.force_keys.add(key)
        self.suppress_keys.discard(key)
        return True

    def toggle_exclude_channel(self, channel: str) -> bool:
        if channel in self.exclude_channels:
            self.exclude_channels.discard(channel)
            return False
        self.exclude_channels.add(channel)
        return True

    def toggle_exclude_config(self, config: str) -> bool:
        if config in self.exclude_configs:
            self.exclude_configs.discard(config)
            return False
        self.exclude_configs.add(config)
        return True

    def edit_state(self, config: str, amp: str, channel: str) -> dict[str, Any]:
        """What the viewer needs to render the current state of one (config, amp, channel)."""
        key = (config, amp, channel)
        forced = None
        if key in self.force_pos_p1:
            forced = ("pos", self.force_pos_p1[key])
        elif key in self.force_neg_p1:
            forced = ("neg", self.force_neg_p1[key])
        return {
            "forced": forced,
            "suppressed": key in self.suppress_keys,
            "whitelisted": key in self.force_keys,
            "channel_excluded": channel in self.exclude_channels,
            "config_excluded": config in self.exclude_configs,
        }

    # ------------------------------------------------------------------ #
    # Applying to the pipeline
    # ------------------------------------------------------------------ #
    def kwargs(self) -> dict[str, Any]:
        """The subset of settings that run_pipeline takes as keyword arguments."""
        out: dict[str, Any] = {}
        for name, value in self.settings.items():
            lever = BY_NAME.get(name)
            if lever is not None and lever.target == "kwarg":
                out[name] = _coerce(name, value)
        return out

    def apply_to_pipeline(self, P) -> None:
        """Push settings + edits onto the imported src.pipeline module.

        `P` is the module object. pipeline.py binds constants BY VALUE at import
        (`from .constants import ...`), so this — not patching src.constants — is what
        the pipeline actually reads.
        """
        for name, value in self.settings.items():
            lever = BY_NAME.get(name)
            if lever is None:
                continue
            if lever.target == "global":
                val = _coerce(name, value)
                if name == "ARTCHAN":
                    # comma-separated names; empty string = auto-detect
                    val = [c.strip() for c in str(val).split(",") if c.strip()] or None
                setattr(P, name, val)
            elif lever.target == "default":
                _rebind_default(P, lever.owner, name, _coerce(name, value))

        # The force/suppress/exclude system.
        P.SIR_FORCE_POS_P1_AFTER = dict(self.force_pos_p1)
        P.SIR_FORCE_NEG_P1_AFTER = dict(self.force_neg_p1)
        P.SIR_FORCE_KEYS = set(self.force_keys)
        P.SIR_SUPPRESS_KEYS = set(self.suppress_keys)
        P.SIR_EXCLUDE_CHANNELS = set(self.exclude_channels)
        P.SIR_EXCLUDE_CONFIGS = set(self.exclude_configs)

        # Point both modes at a session-local template bank. `_resolve_startstop_template_dir`
        # is looked up as a module global at call time, so replacing it is enough — no edit
        # to src/ needed.
        if self.template_dir is not None:
            tdir = Path(self.template_dir)
            P._resolve_startstop_template_dir = lambda _d=tdir: _d

    # ------------------------------------------------------------------ #
    # Persistence / reproducibility
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path) if self.input_path else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "mode": self.mode,
            "template_dir": str(self.template_dir) if self.template_dir else None,
            "template_only_user": self.template_only_user,
            "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.settings.items()},
            "force_pos_p1": [[list(k), v] for k, v in self.force_pos_p1.items()],
            "force_neg_p1": [[list(k), v] for k, v in self.force_neg_p1.items()],
            "force_keys": [list(k) for k in sorted(self.force_keys)],
            "suppress_keys": [list(k) for k in sorted(self.suppress_keys)],
            "exclude_channels": sorted(self.exclude_channels),
            "exclude_configs": sorted(self.exclude_configs),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        s = cls(
            input_path=Path(d["input_path"]) if d.get("input_path") else None,
            output_dir=Path(d["output_dir"]) if d.get("output_dir") else None,
            mode=d.get("mode", "sir"),
            settings=dict(d.get("settings", {})),
            template_dir=Path(d["template_dir"]) if d.get("template_dir") else None,
            template_only_user=bool(d.get("template_only_user", False)),
        )
        s.force_pos_p1 = {tuple(k): (tuple(v) if isinstance(v, list) else v)
                          for k, v in d.get("force_pos_p1", [])}
        s.force_neg_p1 = {tuple(k): (tuple(v) if isinstance(v, list) else v)
                          for k, v in d.get("force_neg_p1", [])}
        s.force_keys = {tuple(k) for k in d.get("force_keys", [])}
        s.suppress_keys = {tuple(k) for k in d.get("suppress_keys", [])}
        s.exclude_channels = set(d.get("exclude_channels", []))
        s.exclude_configs = set(d.get("exclude_configs", []))
        return s

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "Session":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ #
    def to_runner_script(self) -> str:
        """Emit a standalone runner equivalent to this session.

        Round-trips back to the scripted pipeline: the same file can be re-run headless,
        and the result is identical to what the GUI produced.
        """
        globals_lines: list[str] = []
        for name, value in sorted(self.settings.items()):
            lever = BY_NAME.get(name)
            if lever is None or lever.target != "global":
                continue
            val = _coerce(name, value)
            if name == "ARTCHAN":
                val = [c.strip() for c in str(val).split(",") if c.strip()] or None
            globals_lines.append(f"P.{name} = {val!r}")

        default_lines = [
            f"gui_session.rebind_default(P, {BY_NAME[n].owner!r}, {n!r}, {_coerce(n, v)!r})"
            for n, v in sorted(self.settings.items())
            if BY_NAME.get(n) is not None and BY_NAME[n].target == "default"
        ]

        edit_lines = [
            f"P.SIR_FORCE_POS_P1_AFTER = {dict(self.force_pos_p1)!r}",
            f"P.SIR_FORCE_NEG_P1_AFTER = {dict(self.force_neg_p1)!r}",
            f"P.SIR_FORCE_KEYS = {set(self.force_keys) or set()!r}",
            f"P.SIR_SUPPRESS_KEYS = {set(self.suppress_keys) or set()!r}",
            f"P.SIR_EXCLUDE_CHANNELS = {set(self.exclude_channels) or set()!r}",
            f"P.SIR_EXCLUDE_CONFIGS = {set(self.exclude_configs) or set()!r}",
        ]

        if self.template_dir is not None:
            edit_lines.append(
                "\n# --- session-local template bank (built from a selected response) ---\n"
                f"from pathlib import Path as _Path\n"
                f"P._resolve_startstop_template_dir = lambda: _Path(r\"{self.template_dir}\")"
            )

        kw = self.kwargs()
        kw_lines = [f"    {k}={v!r}," for k, v in sorted(kw.items())]

        return f'''"""Auto-generated by the EMG-SCS-flow GUI. Re-runs this session headlessly.

Every knob below is what the GUI had set; re-running reproduces the same outputs.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.pipeline as P
from gui import session as gui_session
from src import run_pipeline

# --- settings (module globals: pipeline.py binds constants by value at import) ---
{chr(10).join(globals_lines) or "# (none)"}

# --- knobs bound as function defaults (a module-global patch would NOT reach them) ---
{chr(10).join(default_lines) or "# (none)"}

# --- manual edits ---
{chr(10).join(edit_lines)}

out = run_pipeline(
    r"{self.input_path}",
    output_dir=r"{self.output_dir}",
    startstop_mode={self.mode == "startstop"},
{chr(10).join(kw_lines)}
)
print("Outputs saved under:", out)
'''


# Constant name -> the parameter it is bound to as a function default. The names do NOT
# match, so this mapping is load-bearing: getting it wrong fails silently.
#   _run_spontaneous_emg_analysis(..., window_ms=..., hop_ms=..., uv=...)
DEFAULT_ARG_FOR: dict[str, str] = {
    "SPONTANEOUS_EMG_WINDOW_MS": "window_ms",
    "SPONTANEOUS_EMG_ENV_HOP_MS": "hop_ms",
    "SPONTANEOUS_EMG_UV_SCALE": "uv",
}


def rebind_default(P, func_name: str, const_name: str, value: Any) -> None:
    """Change a default argument of a pipeline function in place.

    Needed for SPONTANEOUS_EMG_WINDOW_MS / _ENV_HOP_MS / _UV_SCALE: they are bound as
    defaults of _run_spontaneous_emg_analysis and its caller never passes them, so
    setting the module global has no effect at all.
    """
    fn = getattr(P, func_name, None)
    arg_name = DEFAULT_ARG_FOR.get(const_name)
    if fn is None or arg_name is None or fn.__defaults__ is None:
        return
    names = inspect.getfullargspec(fn).args
    defaults = list(fn.__defaults__)
    offset = len(names) - len(defaults)  # defaults align to the TAIL of the arg list
    if arg_name not in names[offset:]:
        raise RuntimeError(
            f"{func_name} has no default argument {arg_name!r} — pipeline.py changed; "
            "update gui/session.py:DEFAULT_ARG_FOR."
        )
    defaults[names[offset:].index(arg_name)] = value
    fn.__defaults__ = tuple(defaults)


# re-exported so the generated runner script can call it
_rebind_default = rebind_default
