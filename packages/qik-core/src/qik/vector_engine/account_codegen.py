"""Narrow fixed-ABI account event calls; derive bodies from the shared account module."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from typing import Any

from . import account

_EVENT_NAMES = (
    "_f0a_entry",
    "_f0a_exit",
    "_f0a_submit",
    "_f0a_release",
    "_f0a_pending",
    "_f0a_complete",
    "_f0a_clear",
    "_account_order",
    "_account_reserved",
    "_account_fill_reserve",
    "_account_reserve_leg",
    "_account_reserves",
    "_account_clock",
    "_account_submitted",
    "_account_resolve_terminal",
    "_account_attempt",
    "_account_kernel",
)
_BORROWED_ARGS = frozenset(
    (
        "a",
        "economic",
        "orders",
        "view",
        "stop_lane",
        "points_lane",
        "limit_lane",
        "trigger_lane",
        "pair_lanes",
        "fraction_lane",
    )
)


def outline_events(
    loops: list[ast.FunctionDef],
    source: Callable[[Any], ast.FunctionDef],
    rebuild: Mapping[str, str],
    table_size: int,
    orders_slot: int,
) -> list[ast.FunctionDef]:
    """No owning-array tuple crosses an event call; pointer owners stay in the runner."""
    originals = {name: source(getattr(account, name)) for name in _EVENT_NAMES}
    parameters = {name: [arg.arg for arg in node.args.args] for name, node in originals.items()}
    names = {name: "_account_native_" + name.lstrip("_") for name in _EVENT_NAMES}

    class Calls(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.Call:
            self.generic_visit(node)
            if not isinstance(node.func, ast.Name) or node.func.id not in names:
                return node
            name = node.func.id
            if len(node.args) != len(parameters[name]) or node.keywords:
                raise ValueError(f"account event call requires explicit arguments: {name}")
            node.func.id = names[name]
            node.args = [ast.Name("_account_context", ast.Load())] + [
                value
                for arg, value in zip(parameters[name], node.args, strict=True)
                if arg not in _BORROWED_ARGS
            ]
            return node

    class BorrowView(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
            if isinstance(node.value, ast.Name) and node.value.id == "view":
                if node.attr == "event_mode":
                    return ast.Constant(2)  # fixed runner constructs only event-mode-2 contexts
                return ast.Name(
                    {"level_rows": "_account_level_rows", "ev_val": "_account_event_values"}[
                        node.attr
                    ],
                    ast.Load(),
                )
            return node

        def visit_If(self, node: ast.If) -> Any:
            self.generic_visit(node)
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Constant)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.comparators[0], ast.Constant)
            ):
                return node.body if test.left.value == test.comparators[0].value else node.orelse
            return node

    class ScalarStores(ast.NodeTransformer):
        serial = 0

        def visit_Assign(self, node: ast.Assign) -> Any:
            if (
                len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Tuple)
                or not isinstance(node.value, ast.Tuple)
                or not any(isinstance(target, ast.Subscript) for target in node.targets[0].elts)
            ):
                return node
            self.serial += 1
            result = []
            for index, value in enumerate(node.value.elts):
                result.append(
                    ast.Assign([ast.Name(f"_event_rhs_{self.serial}_{index}", ast.Store())], value)
                )
            for index, target in enumerate(node.targets[0].elts):
                result.append(
                    ast.Assign([target], ast.Name(f"_event_rhs_{self.serial}_{index}", ast.Load()))
                )
            return result

    account_view = ast.parse(rebuild["account"], mode="eval").body
    assert isinstance(account_view, ast.Call)
    borrowed = {
        "a": ast.unparse(account_view),
        "economic": rebuild["economic"],
        "orders": f"nb.carray(_pointer(ptrs[{orders_slot}]), (REGISTRY_ROWS, 32), dtype=np.int64)",
        "_account_level_rows": rebuild["level_rows"],
        "_account_event_values": rebuild["ev_val"],
    }
    helpers = []
    for name, node in originals.items():
        node.name = names[name]
        kept = [arg for arg in node.args.args if arg.arg not in _BORROWED_ARGS]
        node.args.args = [ast.arg("_account_context")] + kept
        node.args.defaults = []
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        decorated = ast.parse(
            '@nb.njit(cache=True, inline="never", no_cpython_wrapper=True, no_cfunc_wrapper=True)\n'
            "def event(): pass"
        ).body[0]
        assert isinstance(decorated, ast.FunctionDef)
        node.decorator_list = decorated.decorator_list
        Calls().visit(node)
        BorrowView().visit(node)
        ScalarStores().visit(node)
        reads = {
            part.id
            for part in ast.walk(node)
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load)
        }
        prologue = [f"ptrs = nb.carray(_pointer(_account_context), {table_size}, dtype=np.int64)"]
        prologue.extend(f"{key} = {value}" for key, value in borrowed.items() if key in reads)
        node.body[0:0] = ast.parse("\n".join(prologue)).body
        helpers.append(node)

    for loop in loops:
        Calls().visit(loop)
        body = next(
            part
            for part in loop.body
            if isinstance(part, ast.For) and ast.unparse(part.target) == "i"
        )
        loop.body[loop.body.index(body) : loop.body.index(body)] = ast.parse(
            f"core.feats[{orders_slot}] = _f0a_orders.ctypes.data\n"
            "_account_context = np.int64(core.feats.ctypes.data)"
        ).body
        # Raw-pointer calls cannot retain the registry's NRT owner. A final plain
        # call reads that owner after the loop, preventing an early decref.
        at_return = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Return))
        loop.body[at_return:at_return] = ast.parse("_account_finish(_f0a_state, _f0a_orders)").body
    return helpers
