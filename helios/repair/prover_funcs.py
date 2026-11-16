"""The prover's own code-extraction functions, loaded from its source files without importing
their modules (which pull in pandas, jload, vLLM, ...). Using the exact implementations keeps the
repaired proofs compiled the same way as the original attempts."""
import ast
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _load(path, names):
    src = open(os.path.join(REPO, path), encoding="utf-8").read()
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in nodes}
    if missing:
        raise RuntimeError(f"{path}: functions not found: {missing}")
    ns = {"re": re}
    exec(compile(ast.Module(nodes, []), path, "exec"), ns)
    return ns


_u = _load("prover/utils.py", ["extract_code", "problem_check", "replace_statement_in_proof", "remove_comments",
                               "return_theorem_to_prove", "return_theorem_to_replace", "get_error_str"])
_p = _load("prover/components/prover.py", ["handle"])
extract_code, problem_check, get_error_str, handle = _u["extract_code"], _u["problem_check"], _u["get_error_str"], _p["handle"]


def code_to_compile(lean4_code, response):
    """Model response -> the code string the prover sends to the Lean REPL, or (None, reason)."""
    code = extract_code(response)
    if code == "None" or code is None:
        return None, "no lean block"
    full = problem_check(lean4_code, code)
    if full.startswith("**Error**"):
        return None, full.split("\n")[0][:120]
    return handle(full), None
