"""Interactive terminal UI: `nitrotrace` with no arguments.

A small shell over the CLI commands, standard library only. Slash commands run the same code as
`nitrotrace <command>`; a current case is remembered so `/settle` or `/whatif --gas 382` need no id.
`/cases` (or a bare `/`) opens an arrow-key picker you can filter by typing.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import sys

from . import cli
from .audit import ENGINE_VERSION, run_audit
from .schemas import ExperimentRecord
from .validation import load_records

LOGO = ("█▄ █ █ ▀█▀ █▀█ █▀█ ▀█▀ █▀█ ▄▀█ █▀▀ █▀▀",
        "█ ▀█ █  █  █▀▄ █▄█  █  █▀▄ █▀█ █▄▄ ██▄")
GRADIENT = (51, 45, 39, 38, 44, 43, 42, 48, 84, 120)  # 256-colour cyan → green, one step per letter
ACCENT = "38;5;43"

CASE_COMMANDS = ("audit", "settle", "whatif", "casefile", "review-pack")
COMMANDS = [  # (command, arguments, what it does)
    ("cases", "", "pick a record (type to filter, ↑↓, Enter)"),
    ("audit", "[CASE] [--scale K] [--format md|json]", "verdict, budget and what stays unbounded"),
    ("settle", "[CASE]", "what would close the gap, with assumptions"),
    ("whatif", "[CASE] --gas [SRC=]PPB --blank-nmol N --components N", "hypothetical measurements, before → after"),
    ("casefile", "[CASE] [-o FILE]", "source-linked case file (Markdown)"),
    ("review", "[CASE] [-o FILE]", "independent-review checklist"),
    ("plan", "--rate R --flow SCCM --ppb PPB ...", "design before you run"),
    ("stress", "--rate R --unit UNIT --flow SCCM", "impurity level that alone explains a rate"),
    ("bench", "", "benchmark + machine-checked properties"),
    ("extract", "FILE [--llm]", "find evidence passages in a .txt / .pdf"),
    ("list", "", "all records with their verdicts"),
    ("clear", "", "clear the screen"),
    ("help", "", "this list"),
    ("exit", "", "quit (or Ctrl+D)"),
]
ALIASES = {"review": "review-pack", "quit": "exit", "q": "exit", "?": "help", "ls": "list", "open": "cases"}

_ANSI = re.compile(r"\033\[[0-9;]*[A-Za-z]")


def _vt_enable() -> None:
    """Turn on ANSI escape handling in the classic Windows console (Windows Terminal already has it)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:  # noqa: BLE001 - colours are cosmetic
        pass


def vlen(s: str) -> int:
    return len(_ANSI.sub("", s))


def fg(t: str, code: str) -> str:
    return cli.c(t, code)


def width() -> int:
    return max(48, min(shutil.get_terminal_size((88, 24)).columns - 1, 92))


def box(lines: list[str], w: int, color: str = ACCENT, title: str = "") -> str:
    top = "╭" + (f"─ {title} " if title else "") + "─" * (w - 2 - (vlen(title) + 3 if title else 0)) + "╮"
    out = [fg(top, color)]
    for ln in lines:
        out.append(fg("│", color) + " " + ln + " " * max(0, w - 3 - vlen(ln)) + fg("│", color))
    out.append(fg("╰" + "─" * (w - 2) + "╯", color))
    return "\n".join(out)


def logo_lines() -> list[str]:
    out = []
    for row in LOGO:
        col, s = 0, ""
        for p in re.findall(r"\S+|\s+", row):  # colour each letter by its position
            if p.strip():
                s += fg(p, f"38;5;{GRADIENT[min(col, len(GRADIENT) - 1)]};1")
                col += 1
            else:
                s += p
        out.append(s)
    return out


# ------------------------------------------------------------------ keys ----
def _read_key() -> str:
    """One keypress as 'up', 'down', 'enter', 'esc', 'backspace' or the character."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "")
        if ch == "\x03":
            raise KeyboardInterrupt
        return {"\r": "enter", "\x1b": "esc", "\x08": "backspace"}.get(ch, ch)
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # read the descriptor directly: sys.stdin's buffer would swallow the rest of an escape sequence
        data = os.read(fd, 32).decode("utf-8", "ignore")
        if data.startswith("\x1b"):
            return {"\x1b": "esc", "\x1b[A": "up", "\x1bOA": "up", "\x1b[B": "down", "\x1bOB": "down"}.get(data, "")
        if "\x03" in data:
            raise KeyboardInterrupt
        return {"\r": "enter", "\n": "enter", "\x7f": "backspace", "\x08": "backspace"}.get(data, data)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def pick(title: str, items: list[tuple[str, str, str]], rows: int = 12) -> str | None:
    """Arrow-key picker over (key, label, detail) items with type-to-filter. Returns the key or None."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    query, sel, top, drawn = "", 0, 0, 0
    w = width()
    while True:
        q = query.lower()
        shown = [it for it in items if q in (it[0] + " " + it[1]).lower()]
        sel = min(sel, max(len(shown) - 1, 0))
        top = min(max(top, sel - rows + 1), sel)
        lines = [fg(f"  {title}", "1") + fg(f"   filter: {query}▏" if query else "   type to filter · ↑↓ · Enter · Esc", "2")]
        pad = min(max((vlen(it[1]) for it in shown), default=0), 40)
        for i, (key, label, detail) in enumerate(shown[top:top + rows], start=top):
            mark = fg("❯ ", ACCENT) if i == sel else "  "
            body = ((fg(label, f"{ACCENT};1") if i == sel else label) + " " * max(1, pad + 2 - vlen(label))
                    + fg(detail, "2"))
            ln = "  " + mark + body
            lines.append(ln if vlen(ln) <= w else _ANSI.sub("", ln)[: w - 1] + "…")
        if not shown:
            lines.append(fg("    no match", "2"))
        more = len(shown) - (top + rows)
        lines.append(fg(f"    … {more} more" if more > 0 else "", "2"))
        if drawn:
            sys.stdout.write(f"\033[{drawn}A")
        sys.stdout.write("".join("\r\033[2K" + ln + "\n" for ln in lines))
        # clear any leftover lines from a taller previous frame
        extra = drawn - len(lines)
        if extra > 0:
            sys.stdout.write("\033[2K\n" * extra + f"\033[{extra}A")
        sys.stdout.flush()
        drawn = len(lines)
        k = _read_key()
        if k == "up":
            sel = max(sel - 1, 0)
        elif k == "down":
            sel = min(sel + 1, max(len(shown) - 1, 0))
        elif k == "enter":
            return shown[sel][0] if shown else None
        elif k == "esc":
            return None
        elif k == "backspace":
            query, sel = query[:-1], 0
        elif k and k.isprintable():  # one key, or several typed / pasted at once
            query, sel = query + k, 0


# ------------------------------------------------------------------ shell ---
class Shell:
    def __init__(self) -> None:
        self.records = load_records()
        self.case: str | None = None  # case id shown to the user
        self.ref: str | None = None  # what the CLI resolves: the case id, or a JSON path
        self.status: str | None = None
        self.color = cli._COLOR

    # -- views ---------------------------------------------------------------
    def welcome(self) -> None:
        w = width()
        n_pub = sum(k.startswith("cases/") for k in self.records)
        n_syn = sum(k.startswith("synthetic/") for k in self.records)
        tips = [("/cases", "pick a record"), ("/settle", "what closes the gap"),
                ("/audit ocv", "audit by part of an id"), ("/whatif --gas 382", "try a measurement"),
                ("/plan", "design before you run"), ("/help", "all commands")]
        lines = [""] + ["  " + ln for ln in logo_lines()]
        lines[-1] += "   " + fg(f"v{ENGINE_VERSION}", "2")
        lines += ["", "  " + fg("What contamination remains unbounded, and what would close the gap?", "1"),
                  "  " + fg("Electrochemical NH3: contamination budget, verdict, evidence requirements.", "2"), "",
                  "  " + fg(f"{n_pub} published · {n_syn} synthetic records · deterministic, no AI model in any verdict", "2"),
                  ""]
        two = w >= 90  # two tip columns when they fit, otherwise one
        cell = lambda cmd, what: fg(f"{cmd:<18}", ACCENT) + fg(f"{what:<23}", "2")
        for i in range(0, len(tips), 2 if two else 1):
            lines.append("  " + "  ".join(cell(*t) for t in tips[i:i + (2 if two else 1)]))
        lines.append("")
        print(box(lines, w, title=fg("NitroTrace", "1")))
        print(fg(f"  cwd: {os.getcwd()}", "2"))

    def help(self) -> None:
        cli.rule("Commands")
        for cmd, args, what in COMMANDS:
            print("  " + fg(f"/{cmd:<11}", ACCENT) + " " + what + ("  " + fg(args, "2") if args else ""))
        print(cli.wrap("CASE is any unique part of a record id (e.g. ocv). Without one, the current case is used. "
                       "Plain `nitrotrace <command> ...` syntax works too, and a bare record id opens it."))

    def set_case(self, ref: str, d: dict) -> None:
        self.case, self.ref = d["case_id"], ref
        try:
            self.status = run_audit(ExperimentRecord(**d)).status
        except Exception:  # noqa: BLE001 - an invalid file: the CLI will report it when run
            self.status = None

    def status_line(self) -> str:
        parts = []
        if self.case:
            st = self.status or ""
            parts.append(fg("● ", cli.STATUS_COLOR.get(st, "2")) + self.case + fg(f"  {st.replace('_', ' ')}", "2"))
        else:
            parts.append(fg("no case open · /cases to pick one", "2"))
        parts.append(fg("/ for commands · ? help", "2"))
        return "  " + "   ".join(parts)

    def prompt(self) -> str:
        if not (self.color and sys.stdin.isatty()):
            return input(f"nitrotrace{f' ({self.case})' if self.case else ''} > ")
        w = width()
        print()
        print(fg("╭" + "─" * (w - 2) + "╮", "2"))
        print(fg("│", "2") + " " * (w - 2) + fg("│", "2"))
        print(fg("╰" + "─" * (w - 2) + "╯", "2"))
        print(self.status_line())
        sys.stdout.write("\033[3A\r" + fg("│", "2") + " " + fg("❯", ACCENT) + " ")
        sys.stdout.flush()
        try:
            line = input()
        finally:
            sys.stdout.write("\033[2B\r")  # below the box and the status line
            sys.stdout.flush()
        return line

    # -- dispatch ------------------------------------------------------------
    def pick_case(self) -> bool:
        items = []
        for k, d in self.records.items():
            st = run_audit(ExperimentRecord(**d)).status
            items.append((k, fg("■ ", cli.STATUS_COLOR[st]) + d["case_id"], d["title"][:56]))
        key = pick("Records", items)
        if key is None:
            print(fg("  (no case selected)", "2"))
            return False
        self.set_case(self.records[key]["case_id"], self.records[key])
        return True

    def run_cli(self, argv: list[str]) -> None:
        print(fg("⏺ ", ACCENT) + fg("nitrotrace " + " ".join(shlex.quote(a) for a in argv), "2"))
        try:
            cli.main(argv)
        except SystemExit as e:  # argparse errors, unknown cases, bench failures: report and keep going
            if isinstance(e.code, str):
                print(fg("  " + e.code, "31"))
        except KeyboardInterrupt:
            print(fg("\n  interrupted", "2"))
        except Exception as e:  # noqa: BLE001 - one bad command must not end the session
            print(fg(f"  error: {type(e).__name__}: {e}", "31"))

    def handle(self, line: str) -> bool:
        """Run one input line. Returns False to quit."""
        line = line.strip()
        if not line:
            return True
        if line == "/":
            cmd = pick("Commands", [(c, "/" + c, what) for c, _, what in COMMANDS])
            if cmd is None:
                return True
            line = "/" + cmd
        try:
            argv = shlex.split(line.lstrip("/"), posix=os.name != "nt")
        except ValueError as e:
            print(fg(f"  {e}", "31"))
            return True
        argv = [a.strip('"') for a in argv]
        cmd = ALIASES.get(argv[0], argv[0])
        rest = argv[1:]
        if cmd == "exit":
            return False
        if cmd == "help":
            self.help()
        elif cmd == "clear":
            sys.stdout.write("\033[2J\033[H")
            self.welcome()
        elif cmd == "cases":
            if rest:
                return self.handle("/audit " + " ".join(rest))
            if self.pick_case():
                self.run_cli(["audit", self.ref])
        elif cmd in CASE_COMMANDS:
            positional = [a for a in rest if not a.startswith("-")]
            # options that take a value: their argument is not the case
            takes_value = {"--scale", "--format", "--gas", "--blank", "--blank-nmol", "--components", "--source",
                           "-o", "--output"}
            vals = {rest[i + 1] for i, a in enumerate(rest[:-1]) if a in takes_value}
            positional = [a for a in positional if a not in vals]
            if positional:
                try:
                    key, d = cli.resolve(positional[0])
                    self.set_case(key if key.endswith(".json") and key not in self.records else d["case_id"], d)
                except SystemExit:
                    pass  # unknown or ambiguous: the CLI reports it
            elif self.ref or self.pick_case():
                rest = [self.ref] + rest
            else:
                return True
            self.run_cli([cmd] + rest)
        elif cmd in {c for c, _, _ in COMMANDS}:
            self.run_cli([cmd] + rest)
        else:
            matches = [k for k, d in self.records.items() if argv[0].lower() in d["case_id"].lower()]
            if len(argv) == 1 and len(matches) == 1:
                self.set_case(self.records[matches[0]]["case_id"], self.records[matches[0]])
                self.run_cli(["audit", self.ref])
            else:
                print(fg(f"  Unknown command '{argv[0]}'. Type / for the command palette or ? for help.", "33"))
        return True

    def loop(self) -> None:
        self.welcome()
        armed = False
        while True:
            try:
                line = self.prompt()
                armed = False
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                if armed:
                    print()
                    break
                armed = True
                print(fg("\n  Press Ctrl+C again to exit.", "2"))
                continue
            if not self.handle(line):
                break
        print(fg("  bye — every verdict above is reproducible with `nitrotrace audit <case>`.", "2"))


def run() -> None:
    _vt_enable()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        import readline  # noqa: F401 - line editing and history where available (macOS / Linux)
    except ImportError:
        pass
    Shell().loop()
