"""The suite runs where CI does not: a PATH without uv, and SIGINT inherited as ignored."""

from __future__ import annotations

import ast
import pathlib
import re
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING, final

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_PACKAGES = pathlib.Path(__file__).resolve().parents[2]

# Programs every Unix system has; a test may start them by name.
_SYSTEM_PROGRAMS = frozenset({"git", "sh"})
# On the line that names the program, this comment marks a launch that searches PATH on purpose.
_ON_PURPOSE = "# searches PATH:"
_SHELLS = frozenset({"bash", "dash", "sh", "zsh"})
# Words a shell runs itself, and words after which the next word is a command.
_SHELL_WORDS = frozenset(
    {":", "case", "cd", "done", "echo", "esac", "exit", "false", "fi", "for", "kill", "printf"}
    | {"set", "trap", "true", "wait", "}"}
)
_SHELL_PREFIXES = frozenset(
    {"!", "command", "do", "elif", "else", "env", "exec", "if", "nohup", "then", "time", "until"}
    | {"while", "{"}
)
_SHELL_OPERATORS = ";&|()<>\n"
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=.*", re.DOTALL)
_PATH_CONVERSIONS = frozenset(
    {("builtins", "str"), ("os", "fspath"), ("pathlib", "Path"), ("pathlib", "PurePath")}
)
_EXEC_SUFFIXES = ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")
# How each launcher's argument names its program, the argument's position and its keyword.
# "argv": a sequence, or a string that is the program (a script under shell=True); "exec": the
# positional arguments are the argv; "program": the argument is the program; "command": a
# sequence, or a string that is a script; "shell": a script.
_LAUNCHERS: dict[tuple[str, str], tuple[str, int, str]] = {
    **{
        ("subprocess", name): ("argv", 0, "args")
        for name in ("Popen", "call", "check_call", "check_output", "run")
    },
    ("pty", "spawn"): ("argv", 0, "argv"),
    ("asyncio", "create_subprocess_exec"): ("exec", 0, "program"),
    ("anyio", "open_process"): ("command", 0, "command"),
    ("anyio", "run_process"): ("command", 0, "command"),
    ("asyncio", "create_subprocess_shell"): ("shell", 0, "cmd"),
    ("os", "popen"): ("shell", 0, "cmd"),
    ("os", "system"): ("shell", 0, "command"),
    ("pexpect", "spawn"): ("shell", 0, "command"),
    ("os", "posix_spawn"): ("program", 0, "path"),
    ("os", "posix_spawnp"): ("program", 0, "path"),
    **{("os", f"exec{suffix}"): ("program", 0, "path") for suffix in _EXEC_SUFFIXES},
    **{("os", f"spawn{suffix}"): ("program", 1, "file") for suffix in _EXEC_SUFFIXES},
}

type _Function = ast.FunctionDef | ast.AsyncFunctionDef
type _Elements = list[ast.expr | None]
type _Problem = tuple[ast.expr | None, str]
# (kind, text, node): kind is "text" (a string), "path", "which" (shutil.which) or "unknown".
type _Program = tuple[str, str | None, ast.expr | None]


def _element(elements: _Elements, index: int) -> ast.expr | None:
    """Return ``elements[index]``, or None when it, or an element before it, is unknown."""
    for position, element in enumerate(elements):
        if element is None or position == index:
            return element
    return None


def _string(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _untraced_argv() -> list[tuple[bool, _Elements]]:
    return [(False, [None])]


def _walk_scope(body: list[ast.stmt]) -> Iterator[ast.AST]:
    """Yield the nodes of a scope's body, leaving out the bodies of scopes nested in it."""
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _command_words(script: str) -> list[str]:
    """Return the words a POSIX shell script runs as commands, nested ``sh -c`` scripts too."""
    lexer = shlex.shlex(script, posix=True, punctuation_chars=_SHELL_OPERATORS)
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens = list(lexer)
    words = list[str]()
    expect_command = True
    after_prefix = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        name = token.rsplit("/", 1)[-1]
        if token.startswith("#"):
            while index < len(tokens) and "\n" not in tokens[index]:
                index += 1
        elif token and all(char in _SHELL_OPERATORS for char in token):
            if "<" in token or ">" in token:
                index += 1
            else:
                expect_command = True
        elif not expect_command:
            continue
        elif name in _SHELL_PREFIXES:
            after_prefix = True
        elif _ASSIGNMENT.fullmatch(token) or (after_prefix and token.startswith("-")):
            continue
        else:
            words.append(token)
            expect_command = after_prefix = False
            flag = tokens[index] if index + 1 < len(tokens) else ""
            if name in _SHELLS and flag.startswith("-") and "c" in flag:
                words += _command_words(tokens[index + 1])
                index += 2
    return words


@final
class _LaunchScan:
    """Find the launches in one module whose program is looked up on PATH or cannot be traced.

    A launch passes only when the scan can show it names its program by a path: a string with a
    ``/``, ``sys.executable``, a ``pathlib`` join, or git or sh. It follows names to their
    assignments, a function's parameters to its callers, and ``sh -c`` scripts to their commands.
    """

    def __init__(self, source: str, filename: str) -> None:
        self._lines = source.splitlines()
        self._tree = ast.parse(source, filename)
        self._parents = {
            child: node for node in ast.walk(self._tree) for child in ast.iter_child_nodes(node)
        }
        self._modules = dict[str, str]()
        self._imported = dict[str, tuple[str, str]]()
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name if alias.asname else alias.name.partition(".")[0]
                    self._modules[alias.asname or module] = module
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    self._imported[alias.asname or alias.name] = (node.module, alias.name)

    def problems(self) -> list[str]:
        """Return ``line: problem`` for each launch, at the line of its call."""
        found = dict[str, int]()
        for node in ast.walk(self._tree):
            if not isinstance(node, ast.Call):
                continue
            launcher = self._target(node.func)
            if launcher is None or launcher not in _LAUNCHERS:
                continue
            for culprit, reason in self._launch(node, *_LAUNCHERS[launcher]):
                if _ON_PURPOSE not in self._lines[(culprit or node).lineno - 1]:
                    found.setdefault(f"{node.lineno}: {'.'.join(launcher)} {reason}", node.lineno)
        return sorted(found, key=found.__getitem__)

    def _target(self, func: ast.expr) -> tuple[str, str] | None:
        """Return (module, name) that ``func`` refers to, or None for an expression's attribute."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return (self._modules.get(func.value.id, func.value.id), func.attr)
        if isinstance(func, ast.Name):
            return self._imported.get(func.id, ("builtins", func.id))
        return None

    def _launch(self, call: ast.Call, kind: str, index: int, keyword: str) -> list[_Problem]:
        slot = (
            call.args[index]
            if index < len(call.args)
            else next((kw.value for kw in call.keywords if kw.arg == keyword), None)
        )
        if kind == "exec":
            argvs = self._argvs(ast.List(elts=call.args[index:], ctx=ast.Load()))
        elif kind == "program":
            argvs = [(False, [slot, None])]
        elif slot is None:
            argvs = _untraced_argv()
        else:
            argvs = self._argvs(slot)
        shell_keyword = next((kw.value for kw in call.keywords if kw.arg == "shell"), None)
        shell = kind == "shell" or (
            shell_keyword is not None
            and not (isinstance(shell_keyword, ast.Constant) and not shell_keyword.value)
        )
        problems = list[_Problem]()
        for is_string, elements in argvs:
            if shell or (kind == "command" and is_string):
                problems += self._script(elements, 0)
            else:
                problems += self._argv(elements)
        return problems

    def _argvs(self, expr: ast.expr) -> list[tuple[bool, _Elements]]:
        """Return each argv ``expr`` may hold, marked True where it is a string."""
        if _string(expr) is not None:
            return [(True, [expr])]
        if isinstance(expr, (ast.List, ast.Tuple)):
            argvs: list[_Elements] = [[]]
            for element in expr.elts:
                parts = (
                    [part for _, part in self._argvs(element.value)]
                    if isinstance(element, ast.Starred)
                    else [[element]]
                )
                argvs = [argv + part for argv in argvs for part in parts]
            return [(False, argv) for argv in argvs]
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return [
                (False, left + right)
                for _, left in self._argvs(expr.left)
                for _, right in self._argvs(expr.right)
            ]
        if isinstance(expr, ast.IfExp):
            return self._argvs(expr.body) + self._argvs(expr.orelse)
        if isinstance(expr, ast.Name):
            return [
                argv
                for value in self._values(expr)
                for argv in (self._argvs(value) if value is not None else _untraced_argv())
            ]
        return _untraced_argv()

    def _programs(self, element: ast.expr | None) -> list[_Program]:
        """Return what ``element`` may name as a program."""
        if element is None:
            return [("unknown", None, None)]
        if (text := _string(element)) is not None:
            return [("text", text, element)]
        if isinstance(element, (ast.Name, ast.Attribute)):
            target = self._target(element)
            if target == ("sys", "executable"):
                return [("path", None, element)]
        if isinstance(element, ast.Name) and element.id not in self._imported:
            return [
                program
                for value in self._values(element)
                for program in (
                    self._programs(value) if value is not None else [("unknown", None, element)]
                )
            ]
        if isinstance(element, ast.BinOp) and isinstance(element.op, ast.Div):
            return [("path", None, element)]
        if isinstance(element, ast.JoinedStr) and any(
            "/" in (_string(part) or "") for part in element.values
        ):
            return [("path", None, element)]
        if isinstance(element, ast.Call):
            target = self._target(element.func)
            if target == ("shutil", "which"):
                return [("which", None, element)]
            if target in _PATH_CONVERSIONS and len(element.args) == 1 and not element.keywords:
                return self._programs(element.args[0])
        return [("unknown", None, element)]

    def _argv(self, elements: _Elements) -> list[_Problem]:
        """Return the problems with the program an argv starts."""
        problems = list[_Problem]()
        for kind, text, node in self._programs(_element(elements, 0)):
            if kind == "which":
                problems.append((node, "looks its program up on PATH with shutil.which"))
            elif kind == "unknown":
                problems.append((node, "starts a program the scan cannot trace to a path"))
            elif text is not None:
                name = text.rsplit("/", 1)[-1]
                if "/" not in text and text not in _SYSTEM_PROGRAMS:
                    problems.append((node, f"starts {text!r} from PATH"))
                elif name == "env":
                    problems += self._argv(self._after_env(elements))
                elif name in _SHELLS:
                    problems += self._shell(elements)
        return problems

    def _after_env(self, elements: _Elements) -> _Elements:
        """Return the argv that ``env`` runs: its first word that is no option or assignment."""
        index = 1
        while (word := _string(_element(elements, index))) is not None and (
            word.startswith("-") or "=" in word
        ):
            index += 2 if word in ("-u", "-C", "--unset", "--chdir") else 1
        return elements[index:] if _element(elements, index) is not None else [None]

    def _shell(self, elements: _Elements) -> list[_Problem]:
        option = _element(elements, 1)
        flag = _string(option)
        if flag is None:
            return [(option, "starts a shell with arguments the scan cannot trace")]
        if flag.startswith("-") and "c" in flag:
            return self._script(elements, 2)
        return []

    def _script(self, elements: _Elements, index: int) -> list[_Problem]:
        """Return the problems with the commands of the script at ``elements[index]``, whose
        ``$0`` is the argv that follows it."""
        problems = list[_Problem]()
        for _, text, node in self._programs(_element(elements, index)):
            if text is None:
                problems.append((node, "runs a shell script the scan cannot read"))
                continue
            try:
                words = _command_words(text)
            except ValueError:
                problems.append((node, "runs a shell script the scan cannot parse"))
                continue
            for word in words:
                if word in ("$0", "${0}"):
                    problems += self._argv(elements[index + 1 :])
                elif word.startswith("$") or "`" in word:
                    problems.append((node, f"runs {word!r}, which the scan cannot trace"))
                elif "/" not in word and word not in _SHELL_WORDS | _SYSTEM_PROGRAMS:
                    problems.append((node, f"runs {word!r} from PATH"))
        return problems

    def _values(self, name: ast.Name) -> list[ast.expr | None]:
        """Return each expression ``name`` may hold where it is read; None where untraced."""
        innermost = self._function(name)
        scope = innermost
        while True:
            values = self._assigned(scope, name, before=scope is innermost)
            if scope is not None and name.id in self._parameters(scope):
                values += self._arguments(scope, name.id)
            if values or scope is None:
                return values or [None]
            scope = self._function(scope)

    def _function(self, node: ast.AST) -> _Function | None:
        parent = self._parents.get(node)
        while parent is not None and not isinstance(
            parent, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            parent = self._parents.get(parent)
        return parent

    def _statement(self, node: ast.AST) -> ast.stmt:
        while not isinstance(node, ast.stmt):
            node = self._parents[node]
        return node

    def _assigned(
        self, scope: _Function | None, name: ast.Name, *, before: bool
    ) -> list[ast.expr | None]:
        """Return the values bound to ``name`` in ``scope`` (only those before it if ``before``)."""
        values = list[ast.expr | None]()
        for node in _walk_scope(self._tree.body if scope is None else scope.body):
            if not (isinstance(node, ast.Name) and node.id == name.id):
                continue
            if not isinstance(node.ctx, ast.Store):
                continue
            statement = self._statement(node)
            end = (statement.end_lineno or statement.lineno, statement.end_col_offset or 0)
            if before and end > (name.lineno, name.col_offset):
                continue
            binding = self._parents[node]
            if isinstance(binding, ast.Assign) and node in binding.targets:
                values.append(binding.value)
            elif isinstance(binding, ast.AnnAssign):
                if binding.value is not None:
                    values.append(binding.value)
            else:
                values.append(None)
        return values

    @staticmethod
    def _parameters(function: _Function) -> set[str]:
        arguments = function.args
        every = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
        extra = (arguments.vararg, arguments.kwarg)
        return {arg.arg for arg in every} | {arg.arg for arg in extra if arg is not None}

    def _arguments(self, function: _Function, name: str) -> list[ast.expr | None]:
        """Return what each call of ``function`` elsewhere in the module passes as ``name``."""
        arguments = function.args
        positional = [arg.arg for arg in (*arguments.posonlyargs, *arguments.args)]
        defaults = dict(zip(reversed(positional), reversed(arguments.defaults), strict=False))
        defaults |= {
            arg.arg: default
            for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
            if default is not None
        }
        values = list[ast.expr | None]()
        for call in ast.walk(self._tree):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
                continue
            if call.func.id != function.name or self._function(call) is function:
                continue
            if name in positional and any(isinstance(arg, ast.Starred) for arg in call.args):
                values.append(None)
                continue
            passed = dict(zip(positional, call.args, strict=False))
            passed |= {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
            values.append(passed.get(name, defaults.get(name)))
        return values or [None]


def _helper_path_launches(source: str, filename: str = "test_x.py") -> list[str]:
    """Return ``line: problem`` for each launch in ``source`` that does not name its program by
    a path."""
    return _LaunchScan(source, filename).problems()


def _helper_in_function(*lines: str) -> str:
    return "def _f():\n" + "".join(f"    {line}\n" for line in lines)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('subprocess.run(["uv", "run", "pivot", "repro"])', id="argv-list"),
        pytest.param(
            'subprocess.Popen(\n    [\n        "uv",\n        "run",\n    ]\n)', id="multiline"
        ),
        pytest.param("subprocess.run(['pivot', 'repro'])", id="single-quotes"),
        pytest.param('subprocess.run(["pivot"])', id="one-element-list"),
        pytest.param('subprocess.run(["uv"] + ["run", "pivot"])', id="concatenation"),
        pytest.param('subprocess.run(("uv", "run", "pivot", "repro"))', id="tuple"),
        pytest.param('subprocess.run("uv run pivot repro", shell=True)', id="shell-string"),
        pytest.param('await asyncio.create_subprocess_exec("uv", "run")', id="asyncio-exec"),
        pytest.param('await asyncio.create_subprocess_shell("pivot repro")', id="asyncio-shell"),
        pytest.param('subprocess.run([shutil.which("pivot"), "repro"])', id="shutil-which"),
        pytest.param('PIVOT = "pivot"\nsubprocess.run([PIVOT, "repro"])', id="variable"),
        pytest.param('subprocess.run(["python3.13", "-m", "pivot"])', id="python-version"),
        pytest.param('subprocess.run(["pytest", "-x"])', id="pytest"),
        pytest.param('subprocess.run(["/usr/bin/env", "pivot", "repro"])', id="env"),
        pytest.param('subprocess.run(["sh", "-c", "uv run pivot repro"])', id="sh-c"),
        pytest.param('subprocess.run(["sh", "-c", \'exec "$0"\', "uv"])', id="sh-c-dollar-0"),
        pytest.param('os.execvp("uv", ("uv", "run", "pivot"))', id="os-execvp"),
        pytest.param('os.system("uv run pivot")', id="os-system"),
        pytest.param('subprocess.run(["python", "-c", "print(1)"])', id="python"),
        pytest.param('await anyio.run_process(["pivot", "repro"])', id="anyio-run-process"),
        pytest.param('from subprocess import run\nrun(["uv", "run"])', id="imported-function"),
        pytest.param('import subprocess as sp\nsp.run(["uv", "run"])', id="module-alias"),
        pytest.param("command = make()\nsubprocess.run(command)", id="unresolved-argv"),
        pytest.param('subprocess.run(["env", "X=1", "uv", "run"])', id="env-assignment"),
        pytest.param(
            'subprocess.run(["sh", "-c", "cd /x && X=1 pivot repro"])', id="sh-c-after-cd"
        ),
        pytest.param('subprocess.run(["sh", "-c", "exec $1", "sh", "uv"])', id="sh-c-variable"),
        pytest.param('subprocess.run(["sh", "-c", "sh -c \'uv run\'"])', id="nested-sh-c"),
        pytest.param('pexpect.spawn("uv run pivot")', id="pexpect"),
        pytest.param('os.execv("uv", ["uv", "run"])', id="os-execv-relative"),
        pytest.param('subprocess.run(["pivot"] if fast else [sys.executable])', id="conditional"),
    ],
)
def test_scan_finds_a_launch_that_searches_path(source: str) -> None:
    assert _helper_path_launches(_helper_in_function(*source.splitlines())), source


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('subprocess.run([sys.executable, "-c", "1"])', id="sys-executable"),
        pytest.param(
            'subprocess.run([str(pathlib.Path(sys.executable).parent / "pivot"), "repro"])',
            id="venv-pivot",
        ),
        pytest.param('subprocess.run(["git", "init", "-q"])', id="git"),
        pytest.param('subprocess.run(["/bin/sh", "-c", "printf ok"])', id="sh-script"),
        pytest.param('LOGGERS = ["pivot", "pivot.project"]', id="list-of-names"),
        pytest.param('runner.invoke(cli, ["pivot", "--help"])', id="cli-runner"),
        pytest.param('subprocess.run([os.fspath(venv / "bin" / "pivot")])', id="fspath-join"),
        pytest.param('subprocess.run([f"{venv}/bin/pivot", "init"])', id="f-string-path"),
        pytest.param('subprocess.run(["/usr/bin/env", "-u", "X", "/bin/true"])', id="env-path"),
        pytest.param(
            'subprocess.run(["sh", "-c", "trap \'\' INT; exec \\"$0\\"", sys.executable])',
            id="sh-c-dollar-0-path",
        ),
        pytest.param(
            'uv = shutil.which("uv")  # searches PATH: CI installs uv\nsubprocess.run([uv, "build"])',
            id="marked-on-purpose",
        ),
        pytest.param(
            'PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")\n'
            + "command = [PIVOT, 'repro']\n"
            + "command = ['sh', '-c', 'trap \"\" INT; exec \"$0\" \"$@\"', *command]\n"
            + "subprocess.Popen(command)",
            id="argv-variable-reassigned",
        ),
    ],
)
def test_scan_passes_a_launch_that_names_its_program(source: str) -> None:
    assert not _helper_path_launches(_helper_in_function(*source.splitlines())), source


def test_scan_follows_an_argv_parameter_to_its_callers() -> None:
    source = "\n".join(
        [
            "def _run(args):",
            "    return subprocess.run(args)",
            "",
            "def test_it(venv):",
            '    _run([str(venv / "bin" / "pivot"), "init"])',
            '    _run(["uv", "build"])',
        ]
    )
    assert [line.split(":")[0] for line in _helper_path_launches(source)] == ["2"]


def test_scan_follows_a_module_constant_into_a_function() -> None:
    source = "\n".join(
        [
            '_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")',
            '_UV = "uv"',
            "",
            "def test_it():",
            '    subprocess.run([_PIVOT_BIN, "repro"])',
            '    subprocess.run([_UV, "run"])',
        ]
    )
    assert _helper_path_launches(source) == ["6: subprocess.run starts 'uv' from PATH"]


def test_tests_start_programs_without_searching_path() -> None:
    """Every test starts a program by its path (pivot as ``pathlib.Path(sys.executable).parent
    / "pivot"``), so a shell whose PATH has neither uv nor the venv runs the suite as CI does."""
    sources = sorted(_PACKAGES.glob("*/tests/**/*.py"))
    assert sources, f"no test sources under {_PACKAGES}"
    found = [
        f"{path.relative_to(_PACKAGES)}:{problem}"
        for path in sources
        for problem in _helper_path_launches(
            path.read_text("utf-8"), str(path.relative_to(_PACKAGES))
        )
    ]
    assert not found, f"these launches find their program on PATH: {found}"


def test_ctrl_c_tests_pass_when_pytest_starts_with_sigint_ignored(tmp_path: pathlib.Path) -> None:
    """A background job of a non-interactive shell (``cmd &`` in a script) starts with SIGINT
    ignored, and ``uv run`` passes that on; the suite still interrupts the pivot it starts."""
    nodeid = (
        "packages/pivot/tests/cli/test_worker_termination.py"
        + "::test_ctrl_c_in_watch_mode_lets_the_running_stage_clean_up"
    )
    proc = subprocess.run(
        [
            "sh",
            "-c",
            'trap "" INT; exec "$0" "$@"',
            sys.executable,
            "-m",
            "pytest",
            nodeid,
            "-p",
            "no:cacheprovider",
            f"--basetemp={tmp_path / 'basetemp'}",
        ],
        cwd=_PACKAGES.parent,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"{nodeid} failed with SIGINT ignored:\n{proc.stdout[-3000:]}"
