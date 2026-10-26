"""Internal source binding for Phase 2.2; generated modules own their disk cache.

Emit the shared matcher source, never a separately maintained matching algorithm.
The strategy is a module global throughout the call graph, not a callback argument.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import textwrap
from pathlib import Path
from typing import Any, cast

from . import account, compiled, kernel, protection, straddle, types


def write_bound_module(
    destination: Path, strategy_module: str, strategy_name: str, *, records: bool = False
) -> None:
    """Write an importable internal runner for ``step(state, params, feats, bar, pos)``.

    ``run`` has the prototype _run ABI without its first callback argument. State
    is a structured array with one independent record per parameter row. This is
    a code-generation primitive, not the public strategy decorator or Result API.
    Regenerate when engine or strategy source changes, before importing the module.
    records=True transports typed features with an int64 clock and appends final
    copied state rows to the raw output. The prototype ABI stays unchanged otherwise.
    """
    if not all(part.isidentifier() for part in strategy_module.split(".")):
        raise ValueError("strategy_module must be an importable module name")
    if not strategy_name.isidentifier():
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

        def visit_Call(self, node: ast.Call) -> ast.expr:
            if isinstance(node.func, ast.Name) and node.func.id == "_venue_driver":
                return ast.Name("_venue_column", ast.Load())
            self.generic_visit(node)
            return node

    adapter = ast.parse(_ADAPTER_SOURCE).body[0]
    assert isinstance(adapter, ast.FunctionDef)
    adapter.name = "_bound_step"
    if records:
        # Keep the clock int64 and each named feature's dtype through the ABI.
        class RecordFeatures(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
                if isinstance(node.value, ast.Name) and node.value.id == "feats":
                    return ast.parse("feats[0][i]", mode="eval").body
                return cast(ast.expr, self.generic_visit(node))

            def visit_Name(self, node: ast.Name) -> ast.expr:
                if node.id == "CompiledBar":
                    return ast.Name("Bar", ast.Load())
                return node

        RecordFeatures().visit(adapter)
        for node in ast.walk(adapter):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_strategy"
            ):
                node.args[2] = ast.parse("_features_type(*feats[1:])", mode="eval").body

    decision = source(kernel._core_decision)
    for node in ast.walk(decision):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ):
            node.func = ast.Name("_bound_step", ast.Load())

    functions: list[ast.stmt] = [adapter, decision]
    for name in (
        "_venue_column_advance",
        "_column_advance",
        "_slots_column_advance",
        "_venue_column",
        "_column",
        "_slots_column",
    ):
        node = source(getattr(kernel, name))
        BindGlobals().visit(node)
        functions.append(node)
    runner = ast.parse(_RUNNER_SOURCE).body[0]
    assert isinstance(runner, ast.FunctionDef)
    runner.name = "run"
    for node in ast.walk(runner):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_compiled_"):
            node.attr = node.attr.replace("_compiled_", "_")
    BindGlobals().visit(runner)
    if records:
        for node in runner.body:
            if isinstance(node, ast.Return):
                assert isinstance(node.value, ast.Tuple)
                node.value.elts.append(ast.Name("init_state", ast.Load()))
    functions.append(runner)
    strategy = importlib.import_module(strategy_module)
    fingerprint = hashlib.sha256(
        "\n".join(
            inspect.getsource(module)
            for module in (kernel, compiled, protection, account, straddle, types, strategy)
        ).encode()
    ).hexdigest()
    imports = ", ".join(
        sorted(
            {
                node.id
                for function in functions
                for node in ast.walk(function)
                if isinstance(node, ast.Name) and node.id in vars(kernel)
            }
        )
    )
    header = (
        "# Generated by qik.vector_engine.bound; regenerate before importing.\n"
        f"# Engine and strategy source: {fingerprint}\n"
        f"from {strategy_module} import {strategy_name} as _strategy\n"
        f"from qik.vector_engine.kernel import ({imports})\n"
        "from qik.vector_engine.compiled import RULES_WIDTH, CompiledBar, Position\n"
        "_no_callback = None\n"
    )
    if records:
        header += (
            "from qik.vector_engine.types import Bar, Position\n"
            "_features_type = _strategy.__strategy__.features\n"
            "_strategy = nb.njit(cache=True)(_strategy)\n"
        )
    text = (
        header
        + ast.unparse(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])))
        + "\n"
    )
    if not destination.exists() or destination.read_text() != text:
        destination.write_text(text)


# Scheduler and typed Decision transport; matcher bodies come from kernel above.
_ADAPTER_SOURCE = """@nb.njit(cache=True)
def _bound_step(state, params, i, feats, o, h, lo, c, v, view, out):
    bar = CompiledBar(i, int(feats[i, 0]), o, h, lo, c, v)
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
def _typed_runner_template(init_state, params_grid, feats, o, h, lo, c, volume, days, ends, rules, economic, initial_cash, slots, cores=None, masks=None, session_days=None, expiry_days=None, pair_declared=False):
    init_state = init_state.copy()
    if len(rules) != RULES_WIDTH:
        raise ValueError(f'compiled Rules ABI requires exactly {RULES_WIDTH} values')
    if pair_declared and (not slots):
        raise ValueError('straddle requires the slots scheduler')
    k, n = (len(params_grid), len(c))
    width = 2 * n
    counts = np.empty(k, np.int64)
    meta = np.empty((k, 8))
    a0s, a1s = (np.empty((k, width), np.int64), np.empty((k, width), np.int64))
    a2s, a5s = (np.empty((k, width), np.int8), np.empty((k, width), np.int8))
    a3s, a4s, a6s = (np.empty((k, width)), np.empty((k, width)), np.empty((k, width)))
    sides_all, exits_all = (np.zeros((k, n), np.int8), np.zeros((k, n), np.bool_))
    levels_all = np.full((k, 26, n), np.nan)
    bools_all = np.zeros((k, 3, n), np.bool_)
    decisions_all = np.full((k, 31, n), np.nan)
    sizing_risks_all = np.full((k, n), np.nan)
    for column in nb.prange(k):
        j = np.int64(column)
        sides, exits, levels = (sides_all[j], exits_all[j], levels_all[j])
        long_exits, short_exits = (bools_all[j, 1], bools_all[j, 2])
        force_close, force_open = (np.zeros(n, np.int8), np.zeros(n, np.bool_))
        if masks is not None:
            force_close[:] = masks[j, 0]
            force_open[:] = masks[j, 1]
        view = np.full(14, np.nan)
        view[K.V_LAST_EXIT_I] = -1
        decision = np.empty(31)
        account = np.array([0.0, initial_cash, 0.0, -1.0, 0.0])
        core = K.CoreContext(_no_callback, init_state[j], params_grid[j], feats, volume, view, decision, sides, exits, levels, long_exits, short_exits, force_close, force_open, days if session_days is None else session_days, ends, economic, account, int(rules[20]), int(rules[21]), rules[9], int(rules[13]), decisions_all[j], bool(rules[6]), int(rules[22]) + 4 * int(rules[16]) + 8 * int(rules[8]) + 16 * int(rules[25]) + 32 * int(rules[26]) + 64 * int(rules[5]) + 128 * int(rules[28]), sizing_risks_all[j])
        a0, a1, a2, a3, a4, a5, a6 = (a0s[j], a1s[j], a2s[j], a3s[j], a4s[j], a5s[j], a6s[j])
        slip = rules[9] if rules[8] else 0.0
        combine = int(rules[22]) if len(rules) > 22 else 0
        arm_stop = bool(rules[25])
        day_cancel = bool(rules[26])
        if slots:
            result = K._compiled_slots_column(o, h, lo, c, sides, exits, levels[0], levels[1], levels[2], levels[3], levels[4], days, int(rules[0]), bool(rules[1]), bool(rules[2]), bool(rules[3]), bool(rules[4]), bool(rules[5]), bool(rules[6]), int(rules[7]), slip, levels[5], levels[6], levels[7], levels[8], levels[9], bool(rules[15]), bool(rules[16]), bool(rules[17]), bool(rules[18]), bool(rules[19]), bool(rules[10]), int(rules[11]), bool(rules[12]), int(rules[13]), rules[9], long_exits, short_exits, force_close, force_open, bool(rules[14]), False, a0, a1, a2, a3, a4, a5, a6, 0, core, levels[10], combine=int(rules[22]), amend_close=bool(rules[23]), completed_close=bool(rules[24]), arm_stop=arm_stop, day_cancel=day_cancel, expiry_days=expiry_days, straddle=levels[11:23], from_open=levels[23], stop_fracs=levels[24], target_fracs=levels[25], next_bar=bool(rules[27]), exit_and_rearm=bool(rules[28]))
        else:
            result = K._compiled_column(o, h, lo, c, sides, exits, levels[0], levels[1], levels[2], levels[3], levels[4], days, int(rules[0]), bool(rules[1]), bool(rules[2]), bool(rules[3]), bool(rules[4]), bool(rules[5]), bool(rules[6]), int(rules[7]), bool(rules[8]), rules[9], bool(rules[10]), int(rules[11]), bool(rules[12]), int(rules[13]), rules[9], long_exits, short_exits, force_close, force_open, bool(rules[14]), False, a0, a1, a2, a3, a4, a5, a6, 0, core, levels[10], combine, arm_stop, day_cancel, expiry_days)
        counts[j] = -int(account[2]) if account[2] else result[0]
        meta[j, 0] = result[1]
        meta[j, 1] = result[2]
        meta[j, 2] = result[3]
        meta[j, 3] = result[4]
        meta[j, 4] = result[5]
        meta[j, 5] = result[6]
        meta[j, 6] = account[3]
        meta[j, 7] = account[4]
        bools_all[j, 0] = force_open
    return (counts, meta, a0s, a1s, a2s, a3s, a4s, a5s, a6s, sides_all, exits_all, levels_all, bools_all, sizing_risks_all, decisions_all)
"""
