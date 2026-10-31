"""Account event hooks applied to emitted matchers only.

The exact F0a site census still guards the shared matcher sources. Runtime
policy records control entry refusals; no account code-generation families.
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
            "ORDER_SUBMIT": 17,
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
        "_book_loop": "ac977bc3fedae67f568768e6e0b4013d643a281afa2d7d02a8fa5146b28357b1",
        "_slots_loop": "23270d31e06a9e93d905b4de322212fd84ee483e2a746f2aae2dd9c762e85573",
    }
    if digest != pinned[node.name]:
        raise ValueError(f"F0a enclosing AST changed: {node.name}: {digest}")

    def statements(text: str) -> list[ast.stmt]:
        return ast.parse(text).body

    def entry(side: str, price: str, stop: str, key: str, _allow_skip: str) -> list[ast.stmt]:
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
            f"_f0a_complete(_f0a_state, _f0a_orders, {key}, i, _account_fill_reserve(_f0a_state, core.economic, {price}, {stop}, np.nan, {key}), {side})\n"
            f"if not _f0a_entry(_f0a_state, core.economic, {side}, {price}, {stop}, "
            "(core.sizing_risks[cnt] if cnt < len(core.sizing_risks) else np.nan) if core.economic[E_MODE] == MODE_QTY_FEATURE "
            "else core.sizing_risks[0 if len(core.sizing_risks) == 1 else i], "
            f"i, {key}, len(out_entry_i) + 1):\n"
            + "\n".join("    " + line for line in (cleanup + "continue").splitlines())
            + "\nif cnt < len(out_entry_i) + 1:\n    _f0a_state[ACCOUNT_PREFIX + cnt] = _f0a_state[11]"
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

    def reserves(
        key: str,
        ident: str,
        pair: str,
        market: str = "False",
        side: str | None = None,
        quote: str | None = None,
    ) -> str:
        trigger = "entry_limit" if book else "triggers"
        pairs = "_account_view.ev_val" if book else "straddle"
        held_side = side or f"_held_side({'core' if book else 'hv'}, side_sig, {ident})"
        reservation_quote = quote or f"c[max(0, {key})]"
        return (
            f"_account_reserves(_f0a_state, core.economic, _account_view, {key}, {ident}, {pair}, "
            f"{reservation_quote}, entry_stop, stop_pts, entry_limit, {trigger}, {pairs}, "
            f"{held_side}, "
            f"{'entry_stop[:0]' if book else 'stop_fracs'}, {market})"
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
                        ident = ast.unparse(call.args[7])
                        # Only the chase market command has no retained packet
                        # argument. Its source site supplies the saved context
                        # directly, avoiding repeated runtime normalization.
                        if kind == "K_MARKET" and ident == "-1":
                            cash = reserves(key, "vs[VS_PIDENT]", "False", "True", "pend", "c[i]")
                        else:
                            cash = reserves(key, ident, "False")
                        action = (
                            f"if venue:\n    _f0a_pending(_f0a_state, _f0a_orders, {key}, 1, i, {cash})\n"
                            f"else:\n    _f0a_submit(_f0a_state, _f0a_orders, {key}, 1, {cash})"
                        )
                    else:
                        ident, pair = ast.unparse(call.args[6]), ast.unparse(call.args[9])
                        cash = (
                            reserves(key, ident, pair, "True", quote="c[i]")
                            if kind == "_SLOT_CHASE"
                            else reserves(key, ident, pair)
                        )
                        action = f"_f0a_submit(_f0a_state, _f0a_orders, {key}, {mask(ident)} if {pair} else 1, {cash})"
                    result += statements(
                        f"if ok and {condition}:\n"
                        + "\n".join("    " + s for s in action.splitlines())
                    )
                if call.func.id in ("_ring_drop", "_sdrop"):
                    key = ast.unparse(call.args[4 if book else 2])
                    result += statements(f"_f0a_release(_f0a_state, _f0a_orders, {key}, 3, i)")
                if call.func.id == "_pend_load":
                    result += statements(
                        f"_f0a_pending(_f0a_state, _f0a_orders, pend, {mask('pident')} if pend_pair else 1, i, {reserves('pend', 'pident', 'pend_pair')})"
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
                        result += statements(
                            f"_f0a_pending(_f0a_state, _f0a_orders, i, 1, i, {reserves('i', '_held_ident(core, i)', 'False')})"
                        )
                    # Scalar pend=d stages a chase: its subsequent _spush
                    # registers the market replacement with the event quote.
                    elif target == "pend" and value != "d":
                        result += statements(
                            f"_f0a_pending(_f0a_state, _f0a_orders, pend, {mask('pident')} if pend_pair else 1, i, {reserves('pend', 'pident', 'pend_pair')})"
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
                    f"if not filled:\n    _f0a_state[28] = side\n    _f0a_submit(_f0a_state, _f0a_orders, {key}, 1, _account_fill_reserve(_f0a_state, core.economic, px, st, np.nan, {key}))"
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
                    "for _f0a_q in range(6, ACCOUNT_PREFIX):\n    _f0a_prior[_f0a_q] = _f0a_state[_f0a_q]\n"
                    "for _f0a_q in range(32):\n    _f0a_order_prior[0, _f0a_q] = _f0a_orders[0, _f0a_q]\n"
                    "    _f0a_order_prior[1, _f0a_q] = _f0a_orders[1, _f0a_q]\n"
                    "    _f0a_order_prior[2, _f0a_q] = _f0a_orders[2, _f0a_q]\n"
                    "    for _account_row in range(3, REGISTRY_ROWS):\n        _f0a_order_prior[_account_row, _f0a_q] = _f0a_orders[_account_row, _f0a_q]"
                ) + [part]
            if call.func.id == "_row_restore":
                return statements(
                    "for _f0a_q in range(6, ACCOUNT_PREFIX):\n    _f0a_state[_f0a_q] = _f0a_prior[_f0a_q]\n"
                    "for _f0a_q in range(32):\n    _f0a_orders[0, _f0a_q] = _f0a_order_prior[0, _f0a_q]\n"
                    "    _f0a_orders[1, _f0a_q] = _f0a_order_prior[1, _f0a_q]\n"
                    "    _f0a_orders[2, _f0a_q] = _f0a_order_prior[2, _f0a_q]\n"
                    "    for _account_row in range(3, REGISTRY_ROWS):\n        _f0a_orders[_account_row, _f0a_q] = _f0a_order_prior[_account_row, _f0a_q]"
                ) + [part]
            return part

    # Only visit the bar body: initial scalar setup is not an order event.
    loop = next(p for p in node.body if isinstance(p, ast.For) and ast.unparse(p.target) == "i")
    Hooks().visit(loop)
    setup = (
        "_f0a_state = nb.carray(_pointer(core.feats[12]), ACCOUNT_PREFIX + (1 + len(out_entry_i)) * (ORDER_MULTIPLIER * ORDER_WIDTH + 1), dtype=np.float64)\n"
        if callback
        else "_f0a_state = core.account\n"
    )
    node.body[node.body.index(loop) : node.body.index(loop)] = statements(
        setup
        + "_account_view = _held_view(core)\n_f0a_prior = np.empty(ACCOUNT_PREFIX)\n_f0a_orders = np.full((REGISTRY_ROWS, 32), -1, np.int64)\n"
        "_f0a_order_prior = np.empty((REGISTRY_ROWS, 32), np.int64)"
    )
    node.body[node.body.index(loop) : node.body.index(loop)] = statements(
        "_f0a_state[29] = len(out_entry_i) + 1\n_f0a_state[33] = -1\n"
        "_f0a_state[40] = tick\n_f0a_state[41] = tick_round\n"
        "_f0a_state[42] = bracket_anchor\n_f0a_state[43] = combine\n"
        "_f0a_state[44] = int(core.economic[E_ON_REJECT]) & 2"
    )
    # Independently derive actual open entry orders from matcher state at return.
    # This is a per-config diagnostic, never a hold-bar account read or write.
    finish = "if err != ERR_NONE:\n    _account_kernel(_f0a_state, _f0a_orders, err, eb, s, pos)\n    _f0a_expected = 0\nelif core.account[2] != 0:\n    _f0a_clear(_f0a_state, _f0a_orders, n - 1)\n    _f0a_expected = 0\nelse:\n"
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

    class TypedEventCalls(ast.NodeTransformer):
        """One native signature per event, including constant cancel/clear sites."""

        integer_args = {
            "_f0a_entry": (2, 6, 7, 8),
            "_f0a_exit": (2,),
            "_f0a_submit": (2, 3),
            "_f0a_pending": (2, 3, 4),
            "_f0a_release": (2, 3, 4),
            "_f0a_complete": (2, 3, 5),
            "_account_clock": (2, 4, 5),
            "_account_attempt": (1, 2),
            "_account_kernel": (2, 3, 4, 5),
            "_f0a_clear": (2,),
            "_account_fill_reserve": (5,),
            "_account_reserves": (3, 4, 12),
        }

        def visit_Call(self, node: ast.Call) -> ast.Call:
            call = node
            self.generic_visit(call)
            if not isinstance(call.func, ast.Name):
                return call
            name = call.func.id
            if name == "_f0a_pending" and len(call.args) == 5:
                call.args.append(ast.Tuple([ast.Constant(0.0), ast.Constant(0.0)], ast.Load()))
            for index in self.integer_args.get(name, ()):
                call.args[index] = ast.Call(
                    ast.Attribute(ast.Name("np", ast.Load()), "int64", ast.Load()),
                    [call.args[index]],
                    [],
                )
            if name == "_account_attempt":
                call.args[3] = ast.Call(
                    ast.Attribute(ast.Name("np", ast.Load()), "bool_", ast.Load()),
                    [call.args[3]],
                    [],
                )
            if name == "_account_reserves":
                call.args[14] = ast.Call(
                    ast.Attribute(ast.Name("np", ast.Load()), "bool_", ast.Load()),
                    [call.args[14]],
                    [],
                )
                call.args[5] = ast.Call(
                    ast.Attribute(ast.Name("np", ast.Load()), "bool_", ast.Load()),
                    [call.args[5]],
                    [],
                )
            return call

    class EventClocks(ast.NodeTransformer):
        depth = 0
        quote: str | None = None
        close_phase = False
        status = 4
        reason = 6

        def attempt(self) -> Any:
            paired = "False" if book else "cur_pair"
            return statements(
                f"if (s != 0 or {paired}) and _f0a_state[33] != i:\n"
                "    _account_clock(_f0a_state, core.economic, i, c[i], 3, 10)\n"
                f"    {reserves('i', '_held_ident(core, i)', paired)}\n"
                f"    _account_attempt(_f0a_state, i, s, {paired})"
            )

        def prefix(self, part: ast.stmt, expression: ast.AST) -> Any:
            calls = [
                q
                for q in ast.walk(expression)
                if isinstance(q, ast.Call)
                and isinstance(q.func, ast.Name)
                and q.func.id
                in (
                    "_f0a_submit",
                    "_f0a_release",
                    "_f0a_pending",
                    "_f0a_complete",
                    "_f0a_clear",
                    "_f0a_entry",
                    "_f0a_exit",
                    "_account_kernel",
                )
            ]
            if not calls:
                return [part]
            quote = self.quote or ("c[i]" if self.close_phase or not book else "o[i]")
            for call in calls:
                assert isinstance(call.func, ast.Name)
                if call.func.id == "_f0a_entry":
                    quote = ast.unparse(call.args[3])
                elif call.func.id == "_f0a_exit":
                    quote = ast.unparse(call.args[4])
                elif call.func.id == "_f0a_complete":
                    reserve = call.args[4]
                    assert isinstance(reserve, ast.Call)
                    quote = ast.unparse(reserve.args[2])
            return statements(
                f"_account_clock(_f0a_state, core.economic, i, {quote}, {self.status}, {self.reason})"
            ) + [part]

        def visit_Assign(self, node: ast.Assign) -> Any:
            part = node
            # Refusals at an open retain that open's mark, never its later close.
            value = part.value.elts[0] if isinstance(part.value, ast.Tuple) else part.value
            target = ast.unparse(part.targets[0])
            if (
                target in ("(err, eb)", "error")
                and isinstance(value, ast.Name)
                and value.id.startswith("ERR_")
                and value.id != "ERR_NONE"
            ):
                quote = self.quote or ("c[i]" if self.close_phase or not book else "o[i]")
                return statements(
                    f"_account_clock(_f0a_state, core.economic, i, {quote}, 2, 100 + {value.id})"
                ) + [part]
            return self.prefix(part, part.value)

        def visit_Expr(self, node: ast.Expr) -> Any:
            part = node
            if (
                isinstance(part.value, ast.Call)
                and isinstance(part.value.func, ast.Name)
                and part.value.func.id == "_core_decision"
            ):
                self.close_phase = True
            return self.prefix(part, part.value)

        def visit_If(self, node: ast.If) -> Any:
            part = node
            old = self.quote, self.status, self.reason
            test = ast.unparse(part.test)
            if "day_cancel" in test:
                self.quote, self.status, self.reason = "o[i]", 5, 7
            elif "stop_skip" in test:
                self.status, self.reason = 3, 11
            elif test == "oflat":
                self.quote = "o[i]"
            elif "cap" in test and "oi" in test:
                self.quote, self.status, self.reason = "o[i]", 5, 12
            elif test == "kind == _SLOT_PAIR_IOC":
                self.status, self.reason = 5, 12
            elif test == "pos != 0 or pend >= 0":
                self.status, self.reason = 3, 10
            self.generic_visit(part)
            self.quote, self.status, self.reason = old
            return self.prefix(part, part.test)

        def visit_For(self, node: ast.For) -> Any:
            part = node
            old = self.quote
            target = ast.unparse(part.target)
            self.depth += 1
            if target == "phase":
                self.quote = "cx"
            elif target == "m":
                self.quote = "pt"
            self.generic_visit(part)
            if target == "i":
                part.body += self.attempt()
            self.depth -= 1
            self.quote = old
            return part

        def visit_Continue(self, node: ast.Continue) -> Any:
            part = node
            return self.attempt() + [part] if self.depth == 1 else [part]

    # Loop clocks are source-local expressions; the final error site uses eb.
    EventClocks().visit(loop)
    at_return = next(q for q, p in enumerate(node.body) if isinstance(p, ast.Return))
    for statement in node.body[:at_return]:
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "err != ERR_NONE":
            statement.body[:0] = statements(reserves("eb", "_held_ident(core, eb)", "False"))

    TypedEventCalls().visit(node)
