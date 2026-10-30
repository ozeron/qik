"""F0a measurement hooks, applied to emitted matchers only.

This is a spike, not the F0b policy/ledger API. Admission is reachable only by
the private test adapter; ordinary runs always pass the no-check policy.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from typing import Any


def _shape(node: Any) -> Any:
    """Stable AST fields: ignore locations, field ordering and new empty fields."""
    if isinstance(node, ast.AST):
        return [
            type(node).__name__,
            [
                (key, _shape(value))
                for key, value in sorted(ast.iter_fields(node))
                if value is not None and value != []
            ],
        ]
    if isinstance(node, list):
        return [_shape(part) for part in node]
    return node


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def census(node: ast.FunctionDef) -> list[dict[str, str]]:
    """Record statement shapes and enclosing branches, never line selectors."""
    rows: list[dict[str, str]] = []

    def walk(part: ast.AST, parents: tuple[str, ...]) -> None:
        marker = ""
        text = ast.unparse(part)
        if isinstance(part, ast.Assign):
            target = ast.unparse(part.targets[0])
            if target.startswith("(pos, ei, ep,"):
                marker = "ENTRY_ASSIGN"
            elif any(isinstance(x, ast.Name) and x.id == "pend" for x in ast.walk(part.targets[0])):
                marker = "ORDER_SUBMIT_RELEASE"
        if isinstance(part, ast.Call) and isinstance(part.func, ast.Name):
            marker = {
                "_put_trade": "EXIT",
                "_venue_record": "EXIT",
                "_fill_then_protect": "ENTRY_HELPER_EXIT",
                "_row_save": "SAVE",
                "_row_restore": "RESTORE",
                "_ring_push": "ORDER_SUBMIT",
                "_cmd_push": "ORDER_SUBMIT",
                "_spush": "ORDER_SUBMIT",
                "_ring_drop": "ORDER_RELEASE",
                "_sdrop": "ORDER_RELEASE",
                "_entry_quantity": "ENTRY_QUANTITY",
            }.get(part.func.id, marker)
        if marker:
            rows.append(
                dict(
                    marker=marker,
                    statement=text,
                    shape=_digest(_shape(part)),
                    enclosing=_digest(parents),
                )
            )
        if isinstance(part, (ast.If, ast.For, ast.While)):
            parents += (json.dumps(_shape(part)),)
        for child in ast.iter_child_nodes(part):
            walk(child, parents)

    walk(node, ())
    return rows


def instrument(node: ast.FunctionDef, callback: bool) -> None:
    """Inject event calls; loop sources and callback records stay unchanged."""
    book = node.name == "_book_loop"
    before = Counter(row["marker"] for row in census(node))
    expected = {
        "_book_loop": {
            "ENTRY_ASSIGN": 7,
            "ENTRY_HELPER_EXIT": 2,
            "EXIT": 10,
            "ORDER_SUBMIT_RELEASE": 17,
            "SAVE": 1,
            "RESTORE": 1,
            "ORDER_SUBMIT": 16,
            "ORDER_RELEASE": 1,
            "ENTRY_QUANTITY": 7,
        },
        "_slots_loop": {
            "ENTRY_ASSIGN": 2,
            "EXIT": 6,
            "ORDER_SUBMIT_RELEASE": 17,
            "ORDER_SUBMIT": 17,
            "ORDER_RELEASE": 3,
            "ENTRY_QUANTITY": 2,
        },
    }
    # The census is deliberately exact: changed sources require a reviewed registry.
    if dict(before) != expected[node.name]:
        raise ValueError(f"F0a site census changed: {node.name}: {dict(before)}")
    digest = _digest([(r["marker"], r["shape"], r["enclosing"]) for r in census(node)])
    pinned = {
        "_book_loop": "04046e4fcc092313f79d302ce53e4719d6b10fb61dd435ce6c9b7700370c3bbd",
        "_slots_loop": "23270d31e06a9e93d905b4de322212fd84ee483e2a746f2aae2dd9c762e85573",
    }
    if digest != pinned[node.name]:
        raise ValueError(f"F0a enclosing AST changed: {node.name}: {digest}")

    def statements(text: str) -> list[ast.stmt]:
        return ast.parse(text).body

    def entry(side: str, price: str, stop: str, key: str, allow_skip: str) -> list[ast.stmt]:
        cleanup = (
            (
                "pend = zero\n"
                "vs[VS_RESTING] = 0\nvs[VS_CHASING] = 0\nvs[VS_CLOSING] = 0\n"
                "vs[VS_REJECTED] = 0\nvs[VS_PROT] = 0\n"
                "qmask, qhead, qtail = _ring_drop(ring, qmask, qhead, qtail, vs[VS_PD])\n"
            )
            if book
            else (
                "qmask = _sdrop(ring, qmask, pend)\n"
                "pend, resting, chasing, closing, live = unset, False, False, False, False\n"
            )
        )
        return statements(
            f"_f0a_complete(_f0a_state, _f0a_orders, {key}, i)\n"
            f"if not _f0a_entry(_f0a_state, core.economic, {side}, {price}, {stop}, "
            "(core.sizing_risks[cnt] if cnt < len(core.sizing_risks) else np.nan) if core.economic[E_MODE] == MODE_QTY_FEATURE "
            "else core.sizing_risks[0 if len(core.sizing_risks) == 1 else i], "
            f"{allow_skip}):\n"
            + "\n".join("    " + line for line in (cleanup + "continue").splitlines())
        )

    def mask(ident: str) -> str:
        return (
            f"_f0a_mask(_held_pair(hv, straddle, 0, {ident}), "
            f"_held_pair(hv, straddle, 1, {ident}), "
            f"_held_pair(hv, straddle, 6, {ident}), "
            f"_held_pair(hv, straddle, 7, {ident}))"
        )

    resolve = next(
        (
            p
            for p in ast.walk(node)
            if isinstance(p, ast.Assign)
            and ast.unparse(p.targets[0]) == "(stop, target)"
            and isinstance(p.value, ast.Call)
            and "dident" in ast.unparse(p.value)
        ),
        None,
    )

    class Hooks(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> Any:
            part = node
            prefix: list[ast.stmt] = []
            target = ast.unparse(part.targets[0])
            if target.startswith("(pos, ei, ep,"):
                value = part.value
                assert isinstance(value, ast.Tuple)
                side, price = ast.unparse(value.elts[0]), ast.unparse(value.elts[2])
                # The helper entry was already committed before callback/protection.
                if side != "side" and not (
                    side == "s" and ast.unparse(value.elts[3]) == "st" and self.helper_tail
                ):
                    stop = ast.unparse(value.elts[3]) if book else "stop"
                    key = (
                        "vs[VS_PD]"
                        if side == "pend"
                        else "d"
                        if side in ("sd", "fs", "s_hit")
                        else "i"
                    )
                    allow = (
                        "np.isnan(p_lim)"
                        if side == "pend"
                        else "False"
                        if side == "s_hit"
                        else "True"
                    )
                    if side == "fs":
                        allow = "not dpair and (kind == _SLOT_CHASE or (np.isnan(ptrig) and np.isnan(plim)))"
                    prefix = entry(side, price, stop, key, allow)
                    if not book and side == "fs":
                        assert resolve is not None
                        prefix = statements(ast.unparse(resolve)) + prefix
            calls = [
                p
                for p in ast.walk(part.value)
                if isinstance(p, ast.Call) and isinstance(p.func, ast.Name)
            ]
            for call in calls:
                assert isinstance(call.func, ast.Name)
                if call.func.id == "_fill_then_protect":
                    prefix = (
                        []
                        if ast.unparse(call.args[0]) == "side"
                        else entry(
                            ast.unparse(call.args[0]),
                            ast.unparse(call.args[2]),
                            ast.unparse(call.args[3]),
                            "i",
                            "False",
                        )
                    )
                    # A helper may close immediately; settle after its reason is known.
                    return (
                        prefix
                        + [part]
                        + statements(
                            f"if e == ERR_NONE and why != 0:\n"
                            f"    _f0a_exit(_f0a_state, core.economic, {ast.unparse(call.args[0])}, "
                            f"{ast.unparse(call.args[2])}, {ast.unparse(call.args[5])})"
                        )
                    )
            result = prefix + [part]
            for call in calls:
                assert isinstance(call.func, ast.Name)
                if call.func.id in ("_ring_push", "_cmd_push", "_spush"):
                    at = 4 if call.func.id == "_spush" else 5
                    kind_node = call.args[at]
                    kind = ast.unparse(kind_node)
                    # These are source constants (also in conditional kind
                    # expressions), not runtime admission flags. Omit hooks
                    # for commands that cannot submit an entry before Numba
                    # inlines their otherwise-dead held-pair gathers.
                    eligible = (
                        {"K_MARKET", "K_LIMIT", "K_IOC", "K_OPEN_ENTRY"}
                        if book
                        else {"_SLOT_SUBMIT", "_SLOT_REPLACE", "_SLOT_CHASE"}
                    )
                    possible = {
                        part.id
                        for part in ast.walk(kind_node)
                        if isinstance(part, ast.Name)
                        and part.id.startswith("K_" if book else "_SLOT_")
                    }
                    if possible and possible.isdisjoint(eligible):
                        continue
                    condition = (
                        f"({kind}) in (_SLOT_SUBMIT, _SLOT_REPLACE, _SLOT_CHASE)"
                        if not book
                        else f"((venue and ({kind}) in (K_MARKET, K_LIMIT, K_IOC)) or "
                        f"(not venue and ({kind}) == K_OPEN_ENTRY))"
                    )
                    key = ast.unparse(call.args[at + 1])
                    if book:
                        action = (
                            f"if venue:\n    _f0a_pending(_f0a_state, _f0a_orders, {key}, 1, i)\n"
                            f"else:\n    _f0a_submit(_f0a_state, _f0a_orders, {key}, 1)"
                        )
                    else:
                        ident, pair = ast.unparse(call.args[6]), ast.unparse(call.args[9])
                        action = f"_f0a_submit(_f0a_state, _f0a_orders, {key}, {mask(ident)} if {pair} else 1)"
                    result += statements(
                        f"if ok and {condition}:\n"
                        + "\n".join("    " + s for s in action.splitlines())
                    )
                if call.func.id in ("_ring_drop", "_sdrop"):
                    key = ast.unparse(call.args[4 if book else 2])
                    result += statements(f"_f0a_release(_f0a_state, _f0a_orders, {key}, 3, i)")
                if call.func.id == "_pend_load":
                    result += statements(
                        f"_f0a_pending(_f0a_state, _f0a_orders, pend, {mask('pident')} if pend_pair else 1, i)"
                    )
            targets = (
                list(part.targets[0].elts)
                if isinstance(part.targets[0], ast.Tuple)
                else [part.targets[0]]
            )
            values = list(part.value.elts) if isinstance(part.value, ast.Tuple) else [part.value]
            for index, t in enumerate(targets):
                if index >= len(values):
                    continue
                if isinstance(t, ast.Name) and t.id == "pend":
                    value = ast.unparse(values[index])
                    if value in ("zero", "unset"):
                        result += statements("_f0a_pending(_f0a_state, _f0a_orders, -1, 0, i)")
                    elif book:
                        result += statements("_f0a_pending(_f0a_state, _f0a_orders, i, 1, i)")
                    elif target == "pend":
                        result += statements(
                            f"_f0a_pending(_f0a_state, _f0a_orders, pend, {mask('pident')} if pend_pair else 1, i)"
                        )
                if (
                    isinstance(t, ast.Name)
                    and t.id == "qmask"
                    and ast.unparse(values[index]) == "zero"
                ):
                    result += statements("_f0a_clear(_f0a_state, _f0a_orders, i)")
            return result

        helper_tail = False

        def visit_If(self, node: ast.If) -> Any:
            part = node
            # Marketable-limit successful-open tail: entry committed before helper.
            old = self.helper_tail
            if ast.unparse(part.test) == "why != 0":
                self.helper_tail = True
            self.generic_visit(part)
            self.helper_tail = old
            if book and ast.unparse(part.test) == "core is not None and filled":
                key = "p_chase - chase"
                restage = statements(
                    f"if not filled:\n    _f0a_submit(_f0a_state, _f0a_orders, {key}, 1)"
                )
                return restage + entry("side", "px", "st", key, "not filled") + [part]
            test = ast.unparse(part.test)
            if book and test == "kind == K_CANCEL":
                part.body[:0] = statements("_f0a_release(_f0a_state, _f0a_orders, vd, 3, i)")
            if book and test == "not np.isnan(cap) and sd * (oi - cap) > 0":
                part.body[:0] = statements("_f0a_release(_f0a_state, _f0a_orders, d, 3, i)")
            if not book and test == "pos != 0 or pend >= 0":
                part.body[:0] = statements("_f0a_release(_f0a_state, _f0a_orders, d, 3, i)")
            if not book and test == "kind == _SLOT_PAIR_IOC":
                expired = (
                    "int(np.isnan(_held_pair(hv, straddle, 0, dident)) and "
                    "not np.isnan(_held_pair(hv, straddle, 1, dident))) | "
                    "(int(np.isnan(_held_pair(hv, straddle, 6, dident)) and "
                    "not np.isnan(_held_pair(hv, straddle, 7, dident))) << 1)"
                )
                part.body += statements(
                    f"if fs == 0:\n    _f0a_release(_f0a_state, _f0a_orders, d, {expired}, i)"
                )
            return part

        def visit_Expr(self, node: ast.Expr) -> Any:
            part = node
            call = part.value
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                return part
            if call.func.id in ("_put_trade", "_venue_record"):
                return statements(
                    "_f0a_exit(_f0a_state, core.economic, "
                    + ", ".join(ast.unparse(call.args[k]) for k in (4, 5, 6))
                    + ")"
                ) + [part]
            if call.func.id == "_row_save":
                return statements(
                    "for _f0a_q in range(6, 24):\n    _f0a_prior[_f0a_q] = _f0a_state[_f0a_q]\n"
                    "for _f0a_q in range(32):\n    _f0a_order_prior[0, _f0a_q] = _f0a_orders[0, _f0a_q]\n"
                    "    _f0a_order_prior[1, _f0a_q] = _f0a_orders[1, _f0a_q]"
                ) + [part]
            if call.func.id == "_row_restore":
                return statements(
                    "for _f0a_q in range(6, 24):\n    _f0a_state[_f0a_q] = _f0a_prior[_f0a_q]\n"
                    "for _f0a_q in range(32):\n    _f0a_orders[0, _f0a_q] = _f0a_order_prior[0, _f0a_q]\n"
                    "    _f0a_orders[1, _f0a_q] = _f0a_order_prior[1, _f0a_q]"
                ) + [part]
            return part

    # Only visit the bar body: initial scalar setup is not an order event.
    loop = next(p for p in node.body if isinstance(p, ast.For) and ast.unparse(p.target) == "i")
    Hooks().visit(loop)
    setup = (
        "_f0a_state = nb.carray(_pointer(core.feats[12]), 24, dtype=np.float64)\n"
        if callback
        else "_f0a_state = core.account\n"
    )
    node.body[node.body.index(loop) : node.body.index(loop)] = statements(
        setup + "_f0a_prior = np.empty(24)\n_f0a_orders = np.full((2, 32), -1, np.int64)\n"
        "_f0a_order_prior = np.empty((2, 32), np.int64)"
    )
    # Independently derive actual open entry orders from matcher state at return.
    # This is a per-config diagnostic, never a hold-bar account read or write.
    finish = "if err != ERR_NONE or core.account[2] != 0:\n    _f0a_clear(_f0a_state, _f0a_orders, n - 1)\n    _f0a_expected = 0\nelse:\n"
    if book:
        expected = (
            "_f0a_expected = int(pend != 0)\nif not venue:\n"
            "    for _f0a_q in range(ring.shape[1]):\n"
            "        if (qmask >> _f0a_q) & 1 and ring[1, _f0a_q] == K_OPEN_ENTRY:\n"
            "            _f0a_expected += 1"
        )
    else:
        expected = (
            "_f0a_expected = int(pend >= 0)\nif pend >= 0 and pend_pair:\n"
            f"    _f0a_pm = {mask('pident')}\n"
            "    if n - 1 >= pend + latency + 1:\n"
            "        _f0a_pm = int(not np.isnan(_held_pair(hv, straddle, 0, pident))) | (int(not np.isnan(_held_pair(hv, straddle, 6, pident))) << 1)\n"
            "    _f0a_expected = (_f0a_pm & 1) + ((_f0a_pm >> 1) & 1)\n"
            "for _f0a_q in range(ring.shape[1]):\n"
            "    if (qmask >> _f0a_q) & 1 and int(ring[RW_KIND, _f0a_q]) == _SLOT_REPLACE and int(ring[RW_D, _f0a_q]) != pend:\n"
            f"        _f0a_pm = {mask('int(ring[RW_IDENT, _f0a_q])')} if ring[RW_PAIR, _f0a_q] != 0 else 1\n"
            "        _f0a_expected += (_f0a_pm & 1) + ((_f0a_pm >> 1) & 1)"
        )
    finish += "\n".join("    " + line for line in expected.splitlines())
    finish += "\n_f0a_state[23] = _f0a_expected"
    at_return = next(q for q, p in enumerate(node.body) if isinstance(p, ast.Return))
    node.body[at_return:at_return] = statements(finish)
