"""Generate the shared compiled matcher and the undecorated Python oracle."""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import textwrap
from pathlib import Path
from typing import Any

from . import engine, kernel


def write_python_module(destination: Path, strategy_module: str, strategy_name: str) -> None:
    """Emit an undecorated matcher for the independent Python execution mode."""
    _write_module(destination, strategy_module, strategy_name)


def write_fixed_module(destination: Path) -> None:
    """Emit the same matcher with a strategy-independent callback boundary."""
    _write_module(destination, None, None)


def _write_module(
    destination: Path,
    strategy_module: str | None,
    strategy_name: str | None,
) -> None:
    callback = strategy_module is None
    if strategy_module is not None and not all(
        part.isidentifier() for part in strategy_module.split(".")
    ):
        raise ValueError("strategy_module must be an importable module name")
    if strategy_name is not None and not strategy_name.isidentifier():
        raise ValueError("strategy_name must be an identifier")

    def source(function: Any) -> ast.FunctionDef:
        original = getattr(function, "py_func", function)
        node = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
        assert isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list:
            assert isinstance(decorator, ast.Call)
            for keyword in decorator.keywords:
                if keyword.arg == "cache":
                    keyword.value = ast.Constant(True)
        return node

    class BindGlobals(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
            if isinstance(node.value, ast.Name) and node.value.id == "K":
                return ast.Name(node.attr.replace("_compiled_", "_"), ast.Load())
            return node

    adapter = ast.parse(_ADAPTER_SOURCE).body[0]
    assert isinstance(adapter, ast.FunctionDef)
    adapter.name = "_python_step"

    for node in ast.walk(adapter):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_strategy"
        ):
            node.args[2] = ast.parse("_features_type(*feats[1:])", mode="eval").body

    decision = source(engine.decision._core_decision)
    for node in ast.walk(decision):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ):
            node.func = ast.Name("_python_step", ast.Load())

    if callback:
        # The matcher remains identical; only the call boundary is indirect.
        for node in ast.walk(decision):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_python_step"
            ):
                node.func = ast.Attribute(ast.Name("core", ast.Load()), "step", ast.Load())
    field_writer = None
    if callback:
        # Borrow pointers as at the callback boundary: passing owning arrays to
        # an outlined scan adds per-bar NRT reference-count traffic.
        first = next(
            index
            for index, statement in enumerate(decision.body)
            if isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "dropped"
        )
        scan = decision.body[first : first + 3]
        assert isinstance(scan[-1], ast.For)
        field_writer = ast.parse("""
@nb.njit(cache=True, inline="never", no_cpython_wrapper=True, no_cfunc_wrapper=True)
def _decision_fields(d_addr, size, levels_addr, height, width, at, stored, declared, account_addr):
    d = nb.carray(_pointer(d_addr), size, dtype=np.float64)
    levels = nb.carray(_pointer(levels_addr), (height, width), dtype=np.float64)
    account = nb.carray(_pointer(account_addr), 6, dtype=np.float64)
    row = 0
""").body[0]
        assert isinstance(field_writer, ast.FunctionDef)
        field_writer.body += copy.deepcopy(scan)
        field_writer.body += ast.parse("return mask, dropped, row").body
        decision.body[first : first + 3] = ast.parse(
            "mask, dropped, row = _decision_fields(d.ctypes.data, len(d), levels.ctypes.data, "
            "levels.shape[0], levels.shape[1], at, stored, declared, account.ctypes.data)"
        ).body

        class CallbackStatus(ast.NodeTransformer):
            def visit_Expr(self, node: ast.Expr) -> ast.stmt | list[ast.stmt]:
                call = node.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "step"
                ):
                    call.args = [
                        ast.parse("np.int64(core.feats.ctypes.data)", mode="eval").body,
                        ast.Name("i", ast.Load()),
                    ]
                    return [
                        ast.Expr(call),
                        ast.If(
                            test=ast.parse("core.account[2] != 0", mode="eval").body,
                            body=[ast.Return()],
                            orelse=[],
                        ),
                    ]
                return node

        CallbackStatus().visit(decision)
        # Repeated tuple-field extraction inflates the IR before Numba's inline
        # pass can eliminate it. Borrow each publisher input once per decision.
        attributes = sorted(
            {
                node.attr
                for node in ast.walk(decision)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "core"
            }
        )

        class DecisionInputs(ast.NodeTransformer):
            def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
                if isinstance(node.value, ast.Name) and node.value.id == "core":
                    return ast.Name(f"_decision_{node.attr}", ast.Load())
                self.generic_visit(node)
                return node

        DecisionInputs().visit(decision)
        decision.body[0:0] = ast.parse(
            "\n".join(f"_decision_{name} = core.{name}" for name in attributes)
        ).body
    functions: list[ast.stmt] = [decision] if callback else [adapter, decision]
    if field_writer is not None:
        functions.insert(0, field_writer)
    # The book loop publishes at three points: the per-bar site calls ``_core_decision``, the
    # two rare ones (a chase cancellation, a resting entry's fill) ``_core_decision_rare``. The
    # fixed module outlines the rare name so the publisher is expanded once, not three times;
    # the undecorated Python module simply binds it to its own copy.
    rare_decision = copy.deepcopy(decision)
    rare_decision.name = "_core_decision_rare"
    for decorator in rare_decision.decorator_list:
        assert isinstance(decorator, ast.Call)
        for keyword in decorator.keywords:
            if keyword.arg == "inline":
                keyword.value = ast.Constant("never")
        decorator.keywords += [
            ast.keyword("no_cpython_wrapper", ast.Constant(True)),
            ast.keyword("no_cfunc_wrapper", ast.Constant(True)),
        ]
    functions.append(rare_decision)
    # The loops own their callback sites and retained-row reads, so they are emitted as written.
    for module, name in (
        (engine.book, "_entry_levels"),
        (engine.book, "_book_loop"),
        (engine.book_slots, "_slots_loop"),
        (engine.single, "_column"),
        (engine.slots, "_slots_column"),
    ):
        node = source(getattr(module, name))
        BindGlobals().visit(node)
        if callback:
            # Only invoke crosses the Python boundary. Internal wide-context
            # wrappers are unused and dominate cold compilation.
            for decorator in node.decorator_list:
                assert isinstance(decorator, ast.Call)
                decorator.keywords += [
                    ast.keyword("no_cpython_wrapper", ast.Constant(True)),
                    ast.keyword("no_cfunc_wrapper", ast.Constant(True)),
                ]
        functions.append(node)
    runner = ast.parse(_RUNNER_SOURCE).body[0]
    assert isinstance(runner, ast.FunctionDef)
    runner.name = "run"
    for node in ast.walk(runner):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_compiled_"):
            node.attr = node.attr.replace("_compiled_", "_")
    BindGlobals().visit(runner)
    for node in runner.body:
        if isinstance(node, ast.Return):
            assert isinstance(node.value, ast.Tuple)
            node.value.elts.append(ast.Name("init_state", ast.Load()))
    if callback:
        decorator = runner.decorator_list[0]
        assert isinstance(decorator, ast.Call)
        decorator.keywords += [
            ast.keyword("no_cpython_wrapper", ast.Constant(True)),
            ast.keyword("no_cfunc_wrapper", ast.Constant(True)),
        ]
        runner.args.args.insert(0, ast.arg(arg="step"))
        for node in ast.walk(runner):
            if isinstance(node, ast.Name) and node.id == "_no_callback":
                node.id = "step"
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "CoreContext"
            ):
                node.args = [
                    ast.Constant(2) if isinstance(arg, ast.Name) and arg.id == "event_mode" else arg
                    for arg in node.args
                ]
    if callback:
        # The context owns no storage; all pointed-to arrays live through run.
        # Build once per column after its view and decision buffers are allocated.
        for node in ast.walk(runner):
            if isinstance(node, ast.For):
                for index, statement in enumerate(node.body):
                    if isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "core"
                        for target in statement.targets
                    ):
                        setup = ast.parse("""
context = np.array([init_state[j].ctypes.data, params_grid[j].ctypes.data,
                    feats[0].ctypes.data, feats[1].ctypes.data,
                    o.ctypes.data, h.ctypes.data, lo.ctypes.data, c.ctypes.data,
                    volume.ctypes.data, view.ctypes.data, decision.ctypes.data, n, account.ctypes.data], dtype=np.int64)
""").body[0]
                        node.body.insert(index, setup)
                        assert isinstance(statement.value, ast.Call)
                        statement.value.args[3] = ast.Name("context", ast.Load())
                        break
    if not callback:
        # The plain Python loop must surface this config's refusal before a later
        # callback can raise. Native prange still carries counts to the boundary.
        for node in ast.walk(runner):
            if isinstance(node, ast.For):
                for index, statement in enumerate(node.body):
                    if (
                        isinstance(statement, ast.Assign)
                        and isinstance(statement.targets[0], ast.Subscript)
                        and isinstance(statement.targets[0].value, ast.Name)
                        and statement.targets[0].value.id == "counts"
                    ):
                        node.body.insert(
                            index + 1,
                            ast.parse(
                                "if counts[j] == -ACCOUNT_INVALID_SIDE:\n"
                                "    raise ValueError('compiled core: side must be -1, 0, 1')"
                            ).body[0],
                        )
                        break
    functions.append(runner)
    strategy = None if callback else importlib.import_module(strategy_module)
    fingerprint = engine.fingerprint(*((strategy,) if strategy is not None else ()))
    names = {
        node.id
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id in vars(kernel)
    }
    header = (
        "# Generated by qik.vector_engine.codegen; regenerate before importing.\n"
        f"# Engine and strategy source: {fingerprint}\n"
        + ("" if callback else f"from {strategy_module} import {strategy_name} as _strategy\n")
        + "\n".join(engine.imports(names))
        + "\n"
        "from qik.vector_engine.compiled import RULES_WIDTH\n"
        "_no_callback = None\n"
    )
    if not callback:
        header += (
            "from qik.vector_engine.types import Bar, Position\n"
            "_features_type = _strategy.__strategy__.features\n"
        )
    else:
        header += "from qik.vector_engine.fixed import _pointer\n"
    if not callback:
        for function in functions:
            if isinstance(function, ast.FunctionDef):
                function.decorator_list = []
    text = (
        header
        + ast.unparse(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])))
        + "\n"
    )
    if not destination.exists() or destination.read_text() != text:
        destination.write_text(text)


# Scheduler and typed Decision transport; matcher bodies come from kernel above.
_ADAPTER_SOURCE = """@nb.njit(cache=True)
def _python_step(state, params, i, feats, o, h, lo, c, v, view, out):
    bar = Bar(i, feats[0][i], o, h, lo, c, v)
    pos = Position(int(view[0]), view[1], int(view[2]), view[3], view[4], view[5], int(view[6]), view[7], int(view[8]), view[9], -1 if np.isnan(view[10]) else int(view[10]), view[11], view[12], view[13])
    decision = _strategy(state, params, feats, bar, pos)
    out[0] = decision[0]
    out[1] = decision[1]
    out[2] = decision[2]
    out[3] = decision[3]
    out[4] = decision[4]
    out[5] = decision[5]
    out[6] = decision[6]
    out[7] = decision[7]
    out[8] = decision[8]
    out[9] = decision[9]
    out[10] = decision[10]
    out[11] = decision[11]
    out[12] = decision[12]
    out[13] = decision[13]
    out[14] = decision[14]
    out[15] = decision[15]
    out[16] = decision[16]
    out[17] = decision[17]
    out[18] = decision[18]
    out[19] = decision[19]
    out[20] = decision[20]
    out[21] = decision[21]
    out[22] = decision[22]
    out[23] = decision[23]
    out[24] = decision[24]
    out[25] = decision[25]
    out[26] = decision[26]
    out[27] = decision[27]
    out[28] = decision[28]
    out[29] = decision[29]
    out[30] = decision[30]

"""

_RUNNER_SOURCE = """@nb.njit(parallel=True, cache=True)
def _typed_runner_template(init_state, params_grid, feats, o, h, lo, c, volume, days, ends, rules, economic, initial_cash, slots, layout, level_rows, store, trades, blank, blank_pairs, risk_row, cores=None, masks=None, session_days=None, expiry_days=None, pair_declared=False, ev_count=None, ev_bar=None, ev_mask=None, ev_val=None, field_rows=None, event_mode=0, risk_store=None):
    init_state = init_state.copy()
    if len(rules) != RULES_WIDTH:
        raise ValueError(f'compiled Rules ABI requires exactly {RULES_WIDTH} values')
    if pair_declared and (not slots):
        raise ValueError('straddle requires the slots scheduler')
    k, n = (len(params_grid), len(c))
    counts = np.empty(k, np.int64)
    meta = np.empty((k, 10))
    a0s, a1s, a2s, a3s, a4s, a5s, a6s = trades
    rows = level_rows
    pairs = rows[11]
    for column in nb.prange(k):
        j = np.int64(column)
        # Buffers are reused across config batches: every stored row starts unvisited.
        levels = store[j]
        levels[:] = np.nan
        scratch = 1 if event_mode == 2 else n
        sides, exits = (np.zeros(scratch, np.int8), np.zeros(scratch, np.bool_))
        long_exits, short_exits = (np.zeros(scratch, np.bool_), np.zeros(scratch, np.bool_))
        force_close = np.zeros(scratch, np.int8)
        force_open = np.full(2 if event_mode == 2 else n, -1 if event_mode == 2 else 0, np.int64)
        external_masks = np.empty((2, 0), np.int8) if masks is None or len(masks) == 0 else masks[j]
        if event_mode != 2 and external_masks.shape[1]:
            force_close[:] = external_masks[0]
            force_open[:] = external_masks[1]
        straddle = levels[pairs:pairs + 12] if pairs >= 0 else blank_pairs
        view = np.full(14, np.nan)
        view[K.V_LAST_EXIT_I] = -1
        decision = np.empty(31)
        # seen trades, equity, refusal code, sizing refusal bar, sizing error code and
        # the bit set of undeclared fields the core emitted
        account = np.array([0.0, initial_cash, 0.0, -1.0, 0.0, 0.0])
        event_count = ev_count[j:j+1] if event_mode else ev_count[0:0]
        risks = risk_store[j] if event_mode == 2 and risk_row >= 0 else K._level_row(levels, blank, risk_row)
        if risk_row >= 0:
            risks[:] = np.nan
        core_days = days if session_days is None or len(session_days) == 0 else session_days
        core = K.CoreContext(_no_callback, init_state[j], params_grid[j], feats, volume, view, decision, sides, exits, levels, long_exits, short_exits, force_close, force_open, core_days, ends, economic, account, int(rules[K.R_FLATTEN]), int(rules[K.R_BOUNDARY]), rules[K.R_TICK], int(rules[K.R_TICK_ROUND]), K._layout_context(layout), bool(rules[K.R_EXIT_NEXT_OPEN]), int(rules[K.R_COMBINE]) + 4 * int(rules[K.R_OTO]) + 8 * int(rules[K.R_ONE_TICK]) + 16 * int(rules[K.R_ARM_STOP]) + 32 * int(rules[K.R_DAY_CANCEL]) + 64 * int(rules[K.R_NEXT_OPEN]) + 128 * int(rules[K.R_EXIT_AND_REARM]), risks, event_count, ev_bar[j], ev_mask[j], ev_val[j], event_mode, np.full(9, -1, np.int64), field_rows, level_rows, np.empty(12), np.empty(10), external_masks)
        core.event_state[6] = 0
        if event_mode == 2:
            for q in (0, 1, 12, 13, 14, 28):
                if field_rows[q] >= 0:
                    ev_val[j, -1, field_rows[q]] = 0.0
        a0, a1, a2, a3, a4, a5, a6 = (a0s[j], a1s[j], a2s[j], a3s[j], a4s[j], a5s[j], a6s[j])
        if slots:
            result = K._compiled_slots_column(o, h, lo, c, sides, exits, K._level_row(levels, blank, rows[0]), K._level_row(levels, blank, rows[1]), K._level_row(levels, blank, rows[2]), K._level_row(levels, blank, rows[3]), K._level_row(levels, blank, rows[4]), days, rules, K._level_row(levels, blank, rows[5]), K._level_row(levels, blank, rows[6]), K._level_row(levels, blank, rows[7]), K._level_row(levels, blank, rows[8]), K._level_row(levels, blank, rows[9]), long_exits, short_exits, force_close, force_open, False, a0, a1, a2, a3, a4, a5, a6, 0, core, K._level_row(levels, blank, rows[10]), expiry_days=expiry_days, straddle=straddle, from_open=K._level_row(levels, blank, rows[23]), stop_fracs=K._level_row(levels, blank, rows[24]), target_fracs=K._level_row(levels, blank, rows[25]))
        else:
            result = K._compiled_column(o, h, lo, c, sides, exits, K._level_row(levels, blank, rows[0]), K._level_row(levels, blank, rows[1]), K._level_row(levels, blank, rows[2]), K._level_row(levels, blank, rows[3]), K._level_row(levels, blank, rows[4]), days, rules, long_exits, short_exits, force_close, force_open, False, a0, a1, a2, a3, a4, a5, a6, 0, core, K._level_row(levels, blank, rows[10]), expiry_days)
        # Negative counts carry refusals out of prange without exceptions.
        counts[j] = -int(account[2]) if account[2] else result[0]
        meta[j, 0] = result[1]
        meta[j, 1] = result[2]
        meta[j, 2] = result[3]
        meta[j, 3] = result[4]
        meta[j, 4] = result[5]
        meta[j, 5] = result[6]
        meta[j, 6] = account[3]
        meta[j, 7] = account[4]
        meta[j, 8] = account[5]
        meta[j, 9] = core.event_state[0] + 1
    return (counts, meta)
"""
