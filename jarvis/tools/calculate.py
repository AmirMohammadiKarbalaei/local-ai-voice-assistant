import ast
import operator


def schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Safely calculate basic maths. Use this for arithmetic, percentages, "
                "simple equations, and numeric calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, for example '250 * 12' or '18 / 3'.",
                    }
                },
                "required": ["expression"],
            },
        },
    }


def calculate(expression: str) -> dict:
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return allowed_operators[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            value = eval_node(node.operand)
            return allowed_operators[type(node.op)](value)

        raise ValueError("Unsupported or unsafe expression.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = eval_node(tree.body)

        return {
            "ok": True,
            "expression": expression,
            "result": result,
        }
    except Exception as e:
        return {
            "ok": False,
            "expression": expression,
            "error": str(e),
        }
