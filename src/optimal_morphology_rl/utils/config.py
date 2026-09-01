"""Config loading helpers with OmegaConf interpolation and runtime context."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


# AST node types allowed in ${eval:...} expressions.  Keeping the whitelist
# tight means config files can do comparisons / arithmetic / ternaries but not
# arbitrary Python calls.
_ALLOWED_EXPR_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.IfExp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Tuple,
    ast.List,
)

# Matches a value that is *entirely* an OmegaConf interpolation.
_INTERPOLATION_RE = re.compile(r"^\$\{(.+)\}$")

# Operator / keyword tokens that indicate an interpolation is an expression
# rather than a simple variable reference or resolver call.
_EXPR_OPERATORS = (
    "==",
    "!=",
    "<=",
    ">=",
    "<",
    ">",
    "and",
    "or",
    "not",
    "in",
    "+",
    "-",
    "*",
    "/",
    "%",
)


def _safe_eval(expr: str, context: dict[str, Any]) -> Any:
    """Evaluate a restricted Python expression against a context dict."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid config expression: {expr!r}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_EXPR_NODES):
            raise ValueError(f"Disallowed syntax in config expression {expr!r}: {type(node).__name__}")

    compiled = compile(tree, filename="<config-eval>", mode="eval")
    return eval(compiled, {"__builtins__": {}}, context)


def _looks_like_expression(text: str) -> bool:
    """Distinguish simple interpolations from inline expressions.

    Examples:
        - ``${mode}`` -> False (interpolation)
        - ``${foo.bar}`` -> False (interpolation)
        - ``${oc.decode:0_1_0}`` -> False (resolver call)
        - ``${mode == "train"}`` -> True (expression)
        - ``${num_envs > 1}`` -> True (expression)
    """
    text = text.strip()
    # Simple interpolation: ${foo} or ${foo.bar}
    if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", text):
        return False
    # Resolver call: ${resolver:args}
    if re.fullmatch(r"[A-Za-z_]\w*:.*", text):
        return False
    # Otherwise treat as expression if it contains any operator-like tokens.
    lowered = text.lower()
    return any(op in lowered for op in _EXPR_OPERATORS)


def _escape_single_quotes(value: str) -> str:
    """Escape backslashes and single quotes for embedding in a quoted string."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _transform_inline_expressions(value: Any) -> Any:
    """Rewrite bare expressions like ${mode == "train"} into ${eval:'...'}."""
    if isinstance(value, dict):
        return {k: _transform_inline_expressions(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_transform_inline_expressions(v) for v in value]
    if isinstance(value, str):
        match = _INTERPOLATION_RE.match(value)
        if match is not None and _looks_like_expression(match.group(1)):
            escaped = _escape_single_quotes(match.group(1))
            return f"${{eval:'{escaped}'}}"
    return value


def _register_eval_resolver(context: dict[str, Any]) -> None:
    """Register the 'eval' resolver bound to the provided context."""

    def _eval(expr: str) -> Any:
        return _safe_eval(expr, context)

    OmegaConf.register_new_resolver("eval", _eval, replace=True)


def load_yaml_with_context(
    path: str | Path,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML file and resolve OmegaConf interpolations.

    Runtime variables passed in ``context`` become available both as
    OmegaConf interpolations (``${mode}``) and inside ``${eval:...}``
    expressions (``${mode == "train"}`` or ``${eval:'mode == "train"'}``).

    Args:
        path: Path to the YAML file.
        context: Mapping of variable names to values exposed to interpolations
            and expressions.  Defaults to an empty mapping.

    Returns:
        A plain Python dict with all interpolations and expressions resolved.
    """
    context = dict(context) if context is not None else {}
    path = Path(path)

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Make inline expressions like ${mode == "train"} usable without requiring
    # the explicit ${eval:'...'} wrapper.
    raw = _transform_inline_expressions(raw)

    # Inject context at the top level so ${mode} style interpolations work out
    # of the box.  File values win over context defaults initially; context is
    # re-added after this merge so CLI-provided context always takes precedence.
    merged: dict[str, Any] = {**raw, **context}

    cfg = OmegaConf.create(merged)
    _register_eval_resolver(context)

    try:
        resolved = OmegaConf.to_container(cfg, resolve=True)
    finally:
        # Avoid leaking this load's context into later config loads.
        if OmegaConf.has_resolver("eval"):
            OmegaConf.clear_resolver("eval")

    # Remove the injected context keys so the returned dict matches the
    # original config structure.
    for key in context:
        resolved.pop(key, None)

    return resolved


def resolve_config(
    config: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve OmegaConf interpolations and expressions in a plain dict.

    This is useful when a config is already in memory (e.g. built from CLI
    overrides) but still contains ``${...}`` placeholders.

    Args:
        config: Configuration dict potentially containing interpolations.
        context: Runtime variables exposed to interpolations and expressions.

    Returns:
        A new plain dict with all interpolations and expressions resolved.
    """
    context = dict(context) if context is not None else {}

    config = _transform_inline_expressions(config)
    merged: dict[str, Any] = {**config, **context}

    cfg = OmegaConf.create(merged)
    _register_eval_resolver(context)

    try:
        resolved = OmegaConf.to_container(cfg, resolve=True)
    finally:
        if OmegaConf.has_resolver("eval"):
            OmegaConf.clear_resolver("eval")

    for key in context:
        resolved.pop(key, None)

    return resolved
