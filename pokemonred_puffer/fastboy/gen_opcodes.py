"""Translate PyBoy's generated opcodes.py (v2.7.0) into C.

Uses the Python AST so operator precedence is preserved exactly; every sub-expression is
fully parenthesized. Local variable types mirror opcodes.pxd (a,b,c,t: int32_t, flag: uint8_t;
DAA: t int, corr uint16_t), so C integer semantics match the Cython build.
Usage: python gen_opcodes.py path/to/pyboy/core/opcodes.py > opcodes_gen.h
"""

import ast
import sys

src = open(sys.argv[1]).read()
tree = ast.parse(src)

CPU_FIELDS = {
    "A",
    "F",
    "B",
    "C",
    "D",
    "E",
    "HL",
    "SP",
    "PC",
    "cycles",
    "halted",
    "bail",
    "stopped",
    "interrupt_master_enable",
    "interrupts_flag_register",
    "interrupts_enabled_register",
}
# registers live in a local CPURegs struct during the run loop
REG_FIELDS = {"A", "F", "B", "C", "D", "E", "HL", "SP", "PC", "cycles"}
MB_FIELDS = {
    "breakpoint_singlestep": "g->breakpoint_singlestep",
    "breakpoint_singlestep_latch": "g->breakpoint_singlestep_latch",
    "cgb": "0",
}
MB_CALLS = {
    "getitem": "RD",
    "setitem": "WR",
    "getitem_io_ports": "RD_IO",
    "setitem_io_ports": "WR_IO",
    "switch_speed": None,
}
CONSTS = {"FLAGC": "4", "FLAGH": "5", "FLAGN": "6", "FLAGZ": "7", "True": "1", "False": "0"}
BINOPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.FloorDiv: "/",
    ast.Mod: "%",
}
CMPOPS = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


def expr(e):
    if isinstance(e, ast.Constant):
        if isinstance(e.value, bool):
            return "1" if e.value else "0"
        assert isinstance(e.value, int), ast.dump(e)
        return str(e.value)
    if isinstance(e, ast.Name):
        return CONSTS.get(e.id, e.id)
    if isinstance(e, ast.Attribute):
        if isinstance(e.value, ast.Name) and e.value.id == "cpu":
            assert e.attr in CPU_FIELDS, e.attr
            return f"r->{e.attr}" if e.attr in REG_FIELDS else f"g->{e.attr}"
        if isinstance(e.value, ast.Attribute) and e.value.attr == "mb":
            return MB_FIELDS[e.attr]
        raise ValueError(ast.dump(e))
    if isinstance(e, ast.BinOp):
        return f"({expr(e.left)} {BINOPS[type(e.op)]} {expr(e.right)})"
    if isinstance(e, ast.UnaryOp):
        op = {ast.Invert: "~", ast.Not: "!", ast.USub: "-"}[type(e.op)]
        return f"({op}{expr(e.operand)})"
    if isinstance(e, ast.Compare):
        assert len(e.ops) == 1
        return f"({expr(e.left)} {CMPOPS[type(e.ops[0])]} {expr(e.comparators[0])})"
    if isinstance(e, ast.BoolOp):
        op = " && " if isinstance(e.op, ast.And) else " || "
        return "(" + op.join(expr(v) for v in e.values) + ")"
    if isinstance(e, ast.IfExp):
        return f"({expr(e.test)} ? {expr(e.body)} : {expr(e.orelse)})"
    if isinstance(e, ast.Call):
        f = e.func
        assert (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "mb"
        ), ast.dump(e)
        name = MB_CALLS[f.attr]
        return f"{name}(g, r, " + ", ".join(expr(a) for a in e.args) + ")"
    raise ValueError(ast.dump(e))


def target(t):
    return expr(t)


def stmts(body, ind):
    out = []
    p = "    " * ind
    for s in body:
        if isinstance(s, ast.Assign):
            assert len(s.targets) == 1
            out.append(f"{p}{target(s.targets[0])} = {expr(s.value)};")
        elif isinstance(s, ast.AugAssign):
            out.append(f"{p}{target(s.target)} {BINOPS[type(s.op)]}= {expr(s.value)};")
        elif isinstance(s, ast.Expr):
            c = s.value
            if (
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name)
                and c.func.value.id == "logger"
            ):
                out.append(f"{p}/* logger call dropped */")
                continue
            if (
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "switch_speed"
            ):
                out.append(f"{p}/* switch_speed: CGB only */")
                continue
            out.append(f"{p}{expr(c)};")
        elif isinstance(s, ast.If):
            out.append(f"{p}if ({expr(s.test)}) {{")
            out += stmts(s.body, ind + 1)
            if s.orelse:
                out.append(f"{p}}} else {{")
                out += stmts(s.orelse, ind + 1)
            out.append(f"{p}}}")
        elif isinstance(s, ast.Return):
            out.append(f"{p}return {expr(s.value) if s.value else '0'};")
        elif isinstance(s, ast.Pass):
            pass
        else:
            raise ValueError(ast.dump(s))
    return out


funcs = {}
dispatch = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        if node.name == "execute_opcode":
            # if/elif chain: opcode == N -> return F(cpu[, v])
            n = node.body[0]
            while True:
                op = n.test.comparators[0].value
                call = n.body[0].value
                dispatch[op] = (call.func.id, len(call.args) == 2)
                if not n.orelse:
                    break
                n = n.orelse[0]
            continue
        funcs[node.name] = node

out = [
    "/* AUTO-GENERATED by gen_opcodes.py from PyBoy v2.7.0 pyboy/core/opcodes.py. Do not edit. */",
    "#pragma once",
]
for name, f in funcs.items():
    args = [a.arg for a in f.args.args]
    sig = "GB *g, CPURegs *r" + (", int v" if "v" in args else "")
    if name == "DAA_27":
        decl = "    int t; uint16_t corr; uint8_t flag;"
    else:
        decl = "    int32_t a, b, c, t; uint8_t flag;"
    body = stmts(f.body, 1)
    used = " ".join(body)
    out.append(f"static inline __attribute__((always_inline)) int op_{name}({sig}) {{")
    out.append(decl + " (void)a_unused_marker;" if False else decl)
    out.append("    (void)flag;" if name != "DAA_27" else "    (void)flag; (void)corr;")
    if name != "DAA_27":
        out.append("    (void)a; (void)b; (void)c; (void)t;")
    out += body
    if not body or not body[-1].strip().startswith("return"):
        out.append("    return 0;")
    out.append("}")


# Direct dispatch on raw bytes: each case decodes its own immediate (same values as
# fetch_and_execute's OPCODE_LENGTHS-based decode: v = pc2 or (pc3 << 8) + pc2, CB ops get v = 0).
lengths = None
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "OPCODE_LENGTHS":
        lengths = [e.value for e in node.value.args[1].elts]
out.append(
    "static inline __attribute__((always_inline)) void dispatch_bytes(GB *g, CPURegs *r, const uint8_t *p) {"
)
out.append("    switch (p[0]) {")
for op in range(256):
    fn, has_v = dispatch[op]
    if op == 0xCB:
        out.append("    case 0xCB:")
        out.append("        switch (p[1]) {")
        for op2 in range(256):
            fn2, has_v2 = dispatch[0x100 + op2]
            assert not has_v2
            out.append(f"        case 0x{op2:02X}: op_{fn2}(g, r); return;")
        out.append("        }")
        out.append("        return;")
        continue
    if has_v:
        L = lengths[op]
        v = "p[1]" if L == 2 else "((p[2] << 8) + p[1])" if L == 3 else "0"
        out.append(f"    case 0x{op:02X}: op_{fn}(g, r, {v}); return;")
    else:
        out.append(f"    case 0x{op:02X}: op_{fn}(g, r); return;")
out.append("    }")
out.append("}")

# Threaded (computed-goto) variant: one label per first byte; each handler ends with NEXT(),
# which re-checks the cpu_tick loop condition and jumps straight to the next handler.
thr = ["/* AUTO-GENERATED by gen_opcodes.py: computed-goto handlers, included inside cpu_tick. */"]
thr.append("#define FB_DISPATCH_TABLE { " + ", ".join(f"&&L_{op:02X}" for op in range(256)) + " }")
thr.append("#define FB_HANDLERS \\")
for op in range(256):
    fn, has_v = dispatch[op]
    if op == 0xCB:
        thr.append("    L_CB: switch (p[1]) { \\")
        for op2 in range(256):
            fn2, _ = dispatch[0x100 + op2]
            thr.append(f"        case 0x{op2:02X}: op_{fn2}(g, &r); break; \\")
        thr.append("    } NEXT(); \\")
        continue
    if has_v:
        L = lengths[op]
        v = "p[1]" if L == 2 else "((p[2] << 8) + p[1])" if L == 3 else "0"
        thr.append(f"    L_{op:02X}: op_{fn}(g, &r, {v}); NEXT(); \\")
    else:
        thr.append(f"    L_{op:02X}: op_{fn}(g, &r); NEXT(); \\")
thr.append("    /* end */")
open(
    sys.argv[1].rsplit("/", 1)[0] + "/../../../../fastboy_threaded_tmp.h"
    if False
    else "opcodes_threaded.h",
    "w",
).write("\n".join(thr) + "\n")

# OPCODE_LENGTHS table
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "OPCODE_LENGTHS":
        vals = [e.value for e in node.value.args[1].elts]
        out.append(
            f"static const uint8_t OPCODE_LENGTHS[{len(vals)}] = {{"
            + ",".join(map(str, vals))
            + "};"
        )
print("\n".join(out))
print(f"/* {len(funcs)} functions, {len(dispatch)} dispatch entries */", file=sys.stderr)
