"""Static module closure and semantic source identity for compiled qik code.

The resolver is shared with the test runtime: globals, import-time factories,
Numba overload implementations and package initializers all participate. Unknown
syntax or dynamic imports widen the closure to the whole tree. Docstrings remain
values: reflection and generated source can observe them; only comments and
formatting are discarded.
"""

from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

# Numba settings that change generated code are part of the namespace; these only change speed
# or location.
CACHE_ENV_IGNORE = frozenset({"NUMBA_CACHE_DIR", "NUMBA_NUM_THREADS"})


def toolchain() -> dict[str, object]:
    """Everything besides the qik sources that a cached kernel depends on.

    Numba's own index checks its version, the Python minor and the target triple/CPU, but not
    NumPy's version; the shared content-keyed lookup must not lean on that, so the namespace
    names all of them explicitly: interpreter, Numba/NumPy/llvmlite and LLVM versions, target
    triple, CPU name and features, and the codegen-relevant NUMBA_* settings.
    """
    from importlib.metadata import version

    import llvmlite.binding as ll  # no Numba/NumPy import: thread limits are not set yet

    ll.initialize_native_target()
    numba_env = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith("NUMBA_") and k not in CACHE_ENV_IGNORE
    }
    if numba_env.get("NUMBA_DISABLE_JIT") == "0":
        del numba_env["NUMBA_DISABLE_JIT"]  # numba's default spelled out: same code, same partition
    return {
        "python": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        **{name: version(name) for name in ("numba", "numpy", "llvmlite")},
        "llvm": ".".join(map(str, ll.llvm_version_info)),
        "triple": ll.get_process_triple(),
        "cpu": ll.get_host_cpu_name(),
        "features": ll.get_host_cpu_features().flatten(),
        "numba_env": numba_env,
    }


def toolchain_namespace(chain: dict[str, object]) -> str:
    """Stable toolchain/flavour partition shared by the runtime and bare engine."""
    return hashlib.sha256(
        json.dumps(["engine-context-v2", chain], sort_keys=True).encode()
    ).hexdigest()[:24]


def machine_cache_root() -> Path:
    """The same private or default root configure() chooses, including compile flags."""
    selected = os.environ.get("NUMBA_CACHE_DIR")
    if selected == os.environ.get("_ALGOTRADING_NUMBA_CACHE_RESOLVED"):
        selected = os.environ.get("_ALGOTRADING_NUMBA_CACHE_ROOT")
    root = Path(selected) if selected else Path.home() / ".cache/algotrading/numba"
    return root / toolchain_namespace(toolchain())


@functools.cache
def semantic_source(source: bytes | str) -> bytes:
    """AST without positions; malformed source remains byte-keyed (fail closed)."""
    try:
        return ast.dump(ast.parse(source), include_attributes=False).encode()
    except (SyntaxError, ValueError):
        return source.encode() if isinstance(source, str) else source


def source_material(source: Path, paths: Sequence[Path]) -> bytes:
    """Semantic closure, or a path-sensitive raw tree when code observes source text.

    A factory may read its own source to produce a constant. AST equivalence then
    does not imply equal globals. Unknown source readers must retain bytes, paths
    and directory membership, including files they read without importing them.
    """
    raw = any(_reads_source(path, source) for path in paths)
    selected = sorted(source.rglob("*.py")) if raw else paths
    parts = [b"raw-tree-v1", str(source.resolve()).encode()] if raw else [b"semantic-v1"]
    for path in selected:
        parts += [
            path.relative_to(source).as_posix().encode(),
            path.read_bytes() if raw else semantic_source(path.read_bytes()),
        ]
    return b"\0".join(parts)


def _reads_source(path: Path, source: Path) -> bool:
    return _source_observer(path.read_bytes(), path.relative_to(source).as_posix())


@functools.cache
def _source_observer(content: bytes, relative: str) -> bool:
    path = Path(relative)
    if relative in ("_cache_identity.py", "vector_engine/reach.py"):
        return False  # these modules compute identities; they never supply native globals
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return True
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def ancestors(node: ast.AST) -> Iterator[ast.AST]:
        while node in parents:
            node = parents[node]
            yield node

    def normalized(node: ast.AST) -> bool:
        return any(
            isinstance(parent, ast.Call)
            and (
                isinstance(parent.func, ast.Name)
                and parent.func.id == "semantic_source"
                or isinstance(parent.func, ast.Attribute)
                and parent.func.attr == "parse"
                and isinstance(parent.func.value, ast.Name)
                and parent.func.value.id == "ast"
            )
            for parent in ancestors(node)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in (
            "co_firstlineno",
            "co_filename",
            "co_linetable",
            "co_lnotab",
            "co_lines",
            "co_positions",
            "lineno",
            "end_lineno",
            "col_offset",
            "currentframe",
            "_getframe",
            "getframeinfo",
        ):
            # Only cache metadata uses this position; native code never sees it.
            statement = parents.get(node)
            if (
                relative == "vector_engine/fixed.py"
                and node.attr == "lineno"
                and any(
                    isinstance(p, ast.FunctionDef) and p.name == "_feature_write_risk"
                    for p in ancestors(node)
                )
            ):
                # This is a refusal message about the strategy's AST, not an
                # engine global. Strategy source remains byte-keyed by reach.
                continue
            if not (
                relative == "vector_engine/engine/__init__.py"
                and isinstance(statement, ast.Assign)
                and ast.unparse(statement) == "self._lineno = py_func.__code__.co_firstlineno"
                and any(
                    isinstance(p, ast.ClassDef) and p.name == "_ThrowawayLocator"
                    for p in ancestors(node)
                )
            ):
                return True
        if isinstance(node, ast.Name) and node.id == "__file__":
            # The engine locates its sources; this exact assignment is cache plumbing.
            statement = next((p for p in ancestors(node) if isinstance(p, ast.Assign)), None)
            if not (
                path.name == "__init__.py"
                and path.parent.name == "engine"
                and isinstance(statement, ast.Assign)
                and ast.unparse(statement) == "_ENGINE_DIR = Path(__file__).resolve().parent"
            ):
                return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                return True
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in (
                    "read_bytes",
                    "read_text",
                    "getsource",
                    "getsourcelines",
                    "getline",
                    "getlines",
                )
                and not normalized(node)
            ):
                if (
                    relative == "vector_engine/engine/__init__.py"
                    and ast.unparse(node) == "path.read_bytes()"
                    and any(
                        isinstance(p, ast.ClassDef) and p.name == "EngineCacheLocator"
                        for p in ancestors(node)
                    )
                ):
                    continue  # verifies a generated module's content-addressed filename
                # These are generated outputs, never qik inputs: codegen compares
                # its destination; both runners read a draft just emitted by it.
                generated_reader = {
                    "vector_engine/codegen.py": "destination.read_text()",
                    "vector_engine/compiled.py": "draft.read_text()",
                    "vector_engine/fixed.py": "draft.read_text()",
                }.get(relative)
                if generated_reader == ast.unparse(node):
                    continue
                # The pinned Numba workaround inspects upstream Numba, covered by
                # the toolchain and its exact source checksum, not qik source.
                if (
                    path.name == "_numba_patches.py"
                    and node.func.attr == "getsource"
                    and ast.unparse(node) == "inspect.getsource(original)"
                    and any(
                        isinstance(p, ast.Assign)
                        and ast.unparse(p) == "original = inline_closurecall.InlineWorker.inline_ir"
                        for p in ast.walk(tree)
                    )
                ):
                    continue
                return True
    return False


@functools.cache
def _tree(content: bytes, filename: str) -> ast.Module:
    """Read-only AST shared by independent resolvers; bytes keep edits observable."""
    return ast.parse(content, filename)


def static_sources(source: Path, roots: Sequence[str]) -> list[Path]:
    """Transitive qik module closure for the named compilation roots."""
    files = sorted(source.rglob("*.py"))
    modules: dict[str, Path] = {}
    trees: dict[str, ast.Module] = {}
    for path in files:
        parts = ("qik", *path.relative_to(source).with_suffix("").parts)
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        modules[name] = path
        try:
            trees[name] = _tree(path.read_bytes(), str(path))
        except (SyntaxError, ValueError):
            return files
    if not set(roots) <= modules.keys():
        return files
    closure = _Closure(modules, trees)
    for name in roots:
        closure.key(name)
    if not closure.run():
        return files
    return sorted(modules[name] for name in closure.keyed | closure.parents)


# Builtins a module may call at import without running qik code; eval/exec/compile can.
_SAFE_BUILTINS = frozenset(dir(builtins)) - {"eval", "exec", "compile", "__import__"}
_Def = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class _Closure:
    """The qik modules whose code can produce values a kernel reads.

    Module-level imports bind globals. Imports inside a function matter only if the function runs
    before compilation (Numba rejects import inside compiled code), so in a keyed module every
    function or class that module scope names is followed: called, decorated or used as a
    decorator, a base class, or handed on by name (a decorator, or Numba for @overload, may call
    it). A function declaring a global is followed too. Following walks the whole body (nested
    functions included) and, transitively, every qik function or class it names, in any module;
    a class is followed with all its methods. Names resolve per lexical scope (a parameter of one
    method does not hide a global in another). A method on, or a call of, a computed value only
    reaches code that was named, and so followed, where the value was made. A decorated class is
    followed only when a decorator is this tree's code. What cannot be
    traced statically (eval/exec/compile, a call to an unbound name such as one from a star
    import) makes its module follow every function it defines.
    """

    def __init__(self, modules: Mapping[str, Path], trees: Mapping[str, ast.Module]) -> None:
        self.modules, self.trees = modules, trees
        self.keyed: set[str] = set()
        self.parents: set[str] = set()
        self.followed: set[tuple[str, str]] = set()
        self.resolved: set[tuple[str, ...]] = set()  # aliases already followed
        self.everything: set[str] = set()  # modules that follow all their functions
        self.pending: list[Callable[[], None]] = []
        self.unclassifiable = False
        self.scopes: dict[str, _Scope] = {}

    def run(self) -> bool:
        while self.pending and not self.unclassifiable:
            self.pending.pop()()
        return not self.unclassifiable

    def scope(self, name: str) -> _Scope:
        """Top-level definitions, import bindings (local -> module, attribute), other names."""
        if name not in self.scopes:
            scope = _Scope({}, {}, set(), {})
            for node in self.trees[name].body:
                if isinstance(node, _Def):
                    scope.definitions[node.name] = node
            for node in _module_level(self.trees[name]):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        scope.bindings[alias.asname or root] = (
                            alias.name if alias.asname else root,
                            None,
                        )
                elif isinstance(node, ast.ImportFrom):
                    base = self.absolute(node, name)
                    for alias in node.names:
                        submodule = f"{base}.{alias.name}"
                        scope.bindings[alias.asname or alias.name] = (
                            (submodule, None) if submodule in self.modules else (base, alias.name)
                        )
                elif isinstance(node, _Def):
                    scope.values.add(node.name)  # class-body definitions (property setters)
                elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                    scope.values.add(node.id)
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and _alias(node.value)
                ):
                    scope.aliases[node.targets[0].id] = node.value
            self.scopes[name] = scope
        return self.scopes[name]

    def absolute(self, node: ast.ImportFrom, name: str) -> str:
        base = node.module or ""
        if not node.level:
            return base
        package = name if self.modules[name].name == "__init__.py" else name.rpartition(".")[0]
        anchor = package.split(".")
        if node.level - 1 >= len(anchor):
            return ""
        return ".".join([*anchor[: len(anchor) - (node.level - 1)], *([base] if base else [])])

    def key(self, name: str) -> None:
        if name in self.modules and name not in self.keyed:
            self.keyed.add(name)
            self.pending.append(lambda: self.module_scope(name))

    def imported(self, node: ast.AST, name: str) -> None:
        """Key what an import statement or dynamic import call in module `name` loads."""
        targets: list[str] = []
        if isinstance(node, ast.Import):
            for alias in node.names:  # import a.b.c binds a; a.b.c is reachable through it
                parts = alias.name.split(".")
                targets += [".".join(parts[:n]) for n in range(1, len(parts) + 1)]
        elif isinstance(node, ast.ImportFrom):
            base = self.absolute(node, name)
            for alias in node.names:
                submodule = f"{base}.{alias.name}"
                targets.append(submodule if submodule in self.modules else base)
        elif isinstance(node, ast.Call):
            argument = node.args[0] if node.args else None
            if not (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and not argument.value.startswith(".")
            ):
                self.unclassifiable = True  # computed or relative: cannot know what loads
                return
            targets.append(argument.value)
        for target in targets:
            self.key(target)
            parts = target.split(".")
            self.parents.update(
                parent
                for parent in (".".join(parts[:n]) for n in range(1, len(parts)))
                if parent in self.modules
            )

    def module_scope(self, name: str) -> None:
        """Follow what runs, or is handed on, when the module executes: calls with their
        arguments, decorators, base classes, defaults and assigned values (a name merely
        mentioned, e.g. in an annotation or __all__, runs nothing)."""
        definitions = self.scope(name).definitions
        running: list[ast.AST] = []
        for node in _module_level(self.trees[name]):
            if isinstance(node, ast.Import | ast.ImportFrom | ast.Call):
                running.append(node)
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
                if node.value is not None and not _alias(node.value):
                    running.append(node.value)
            elif isinstance(node, ast.ClassDef):
                running += [*node.bases, *(keyword.value for keyword in node.keywords)]
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                running += [*node.args.defaults, *filter(None, node.args.kw_defaults)]
            if isinstance(node, _Def):
                running += node.decorator_list
                # A decorator may call a function (Numba calls @overload implementations while
                # compiling); a toolchain class decorator runs no method.
                if definitions.get(node.name) is node and (
                    (node.decorator_list and not isinstance(node, ast.ClassDef))
                    or any(self.internal(decorator, name) for decorator in node.decorator_list)
                    or any(isinstance(inner, ast.Global) for inner in ast.walk(node))
                ):
                    self.follow(name, node.name)
        for expression in running:
            for inner in ast.walk(expression):
                self.visit(inner, name, frozenset())

    def internal(self, expression: ast.expr, name: str) -> bool:
        """Whether a (decorator) expression of module `name` names code of this tree."""
        while isinstance(expression, ast.Call | ast.Attribute):
            expression = expression.func if isinstance(expression, ast.Call) else expression.value
        if not isinstance(expression, ast.Name):
            return True  # computed: assume ours
        scope = self.scope(name)
        if expression.id in scope.bindings:
            return scope.bindings[expression.id][0] in self.modules
        return expression.id in scope.definitions or expression.id in scope.values

    def visit(self, node: ast.AST, name: str, local: frozenset[str]) -> None:
        if isinstance(node, ast.Import | ast.ImportFrom) or _dynamic_import(node):
            self.imported(node, name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            self.reference(node.func.id, [], name, local, call=True)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            self.reference(node.id, [], name, local, call=False)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            chain: list[str] = []
            root: ast.expr = node
            while isinstance(root, ast.Attribute):
                chain.insert(0, root.attr)
                root = root.value
            if isinstance(root, ast.Name):
                self.reference(root.id, chain, name, local, call=False)

    def reference(
        self, identifier: str, chain: list[str], name: str, local: frozenset[str], *, call: bool
    ) -> None:
        """Follow what `identifier.chain` names in module `name`."""
        scope = self.scope(name)
        if identifier in local:
            return
        if identifier in scope.definitions:
            self.follow(name, identifier)
        elif identifier in scope.aliases:
            if (name, identifier, *chain) not in self.resolved:  # also guards alias cycles
                self.resolved.add((name, identifier, *chain))
                target: ast.expr = scope.aliases[identifier]
                path = list(chain)
                while isinstance(target, ast.Attribute):
                    path.insert(0, target.attr)
                    target = target.value
                assert isinstance(target, ast.Name)
                self.reference(target.id, path, name, frozenset(), call=call)
        elif identifier in scope.bindings:
            self.bound(scope.bindings[identifier], chain)
        elif call and identifier not in scope.values and identifier not in _SAFE_BUILTINS:
            self.follow_everything(name)  # eval/exec/compile, or a name nothing binds

    def bound(self, binding: tuple[str, str | None], chain: list[str]) -> None:
        """Follow an import binding plus an attribute chain into this tree's qik modules."""
        target, attribute = binding
        path = [*([attribute] if attribute else []), *chain]
        if target not in self.modules and not any(
            module.startswith(f"{target}.") for module in self.modules
        ):
            return  # the toolchain or another package: not this tree
        while path and f"{target}.{path[0]}" in self.modules:
            target = f"{target}.{path.pop(0)}"
        if path and target in self.modules:
            self.member(target, path[0])

    def member(self, module: str, attribute: str) -> None:
        """Follow `attribute` of qik module `module`."""
        self.key(module)
        scope = self.scope(module)
        if attribute in scope.definitions or attribute in scope.aliases:
            self.reference(attribute, [], module, frozenset(), call=False)
        elif attribute in scope.bindings:
            self.bound(scope.bindings[attribute], [])
        elif attribute not in scope.values:
            self.follow_everything(module)  # e.g. re-exported by a star import

    def follow(self, name: str, definition: str) -> None:
        if (name, definition) in self.followed:
            return
        self.followed.add((name, definition))
        node = self.scope(name).definitions[definition]

        def body() -> None:
            for inner, local in _region([node], frozenset(), frozenset()):
                self.visit(inner, name, local)

        self.pending.append(body)

    def follow_everything(self, name: str) -> None:
        if name not in self.everything:
            self.everything.add(name)
            self.key(name)
            for definition in self.scope(name).definitions:
                self.follow(name, definition)


class _Scope(NamedTuple):
    definitions: dict[str, ast.AST]
    bindings: dict[str, tuple[str, str | None]]
    values: set[str]
    aliases: dict[str, ast.expr]  # NAME = other_name / module.attr: resolved where it is used


_Scoping = (
    ast.FunctionDef
    | ast.AsyncFunctionDef
    | ast.Lambda
    | ast.ClassDef
    | ast.ListComp
    | ast.SetComp
    | ast.DictComp
    | ast.GeneratorExp
)
_Function = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
_Comprehension = ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp


def _region(
    roots: list[ast.AST], here: frozenset[str], down: frozenset[str]
) -> Iterator[tuple[ast.AST, frozenset[str]]]:
    """Nodes evaluated in one scope with the names local there (`here`); nested scopes recurse,
    seeing `down` (a class body's own names are not visible inside its methods)."""
    stack = list(roots)
    while stack:
        node = stack.pop()
        yield node, here
        if isinstance(node, _Scoping):
            stack.extend(_header(node))  # decorators, defaults, bases: the enclosing scope
            inner = down | _bound(node)
            yield from _region(
                _body(node), inner, down if isinstance(node, ast.ClassDef) else inner
            )
        else:
            stack.extend(ast.iter_child_nodes(node))


def _header(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        arguments = node.args
        header: list[ast.AST] = [*arguments.defaults, *filter(None, arguments.kw_defaults)]
        if not isinstance(node, ast.Lambda):
            every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            every += [a for a in (arguments.vararg, arguments.kwarg) if a]
            header += [*node.decorator_list, *(a.annotation for a in every if a.annotation)]
            header += [node.returns] if node.returns else []
        return header
    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases, *(k.value for k in node.keywords)]
    if isinstance(node, _Comprehension):
        return [node.generators[0].iter]
    return []


def _body(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return list(node.body)
    if isinstance(node, ast.Lambda):
        return [node.body]
    if isinstance(node, _Comprehension):
        elements: list[ast.AST] = (
            [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        )
        for index, generator in enumerate(node.generators):
            elements += [generator.target, *generator.ifs]
            elements += [generator.iter] if index else []
        return elements
    return []


def _bound(scope: ast.AST) -> frozenset[str]:
    """Names a scope binds for itself (an under-approximation: a name missed here is followed)."""
    names: set[str] = set()
    declared: set[str] = set()
    if isinstance(scope, _Function):
        arguments = scope.args
        every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        names |= {a.arg for a in [*every, arguments.vararg, arguments.kwarg] if a}
    stack = _body(scope)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Global | ast.Nonlocal):
            declared.update(node.names)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
            continue  # its body is its own scope
        elif isinstance(node, ast.Lambda | _Comprehension):
            continue
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.alias) and node.name != "*":
            names.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        stack.extend(ast.iter_child_nodes(node))
    return frozenset(names - declared)


def _module_level(tree: ast.Module) -> Iterator[ast.AST]:
    """Nodes that run when the module executes: everything outside function bodies, plus
    decorators, defaults and class bodies."""
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            stack += [
                *node.decorator_list,
                *node.args.defaults,
                *filter(None, node.args.kw_defaults),
            ]
        elif isinstance(node, ast.Lambda):
            stack += [*node.args.defaults, *filter(None, node.args.kw_defaults)]
        else:
            stack.extend(ast.iter_child_nodes(node))


def _alias(expression: ast.expr) -> bool:
    """A plain name or attribute chain: binding it runs nothing."""
    while isinstance(expression, ast.Attribute):
        expression = expression.value
    return isinstance(expression, ast.Name)


def _dynamic_import(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
        or (isinstance(node.func, ast.Name) and node.func.id in ("import_module", "__import__"))
    )
