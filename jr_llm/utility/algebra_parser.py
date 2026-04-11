import re
import ast
import operator
import logging

class AlgebraParser:
    def __init__(self):
        self.operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

    def clean_scratch(self, raw: str) -> str:
        cleaned = re.sub(r'[^0-9+\-*/^().]', '', raw)
        return cleaned.replace('^', '**')

    def evaluate_node(self, node):
        if isinstance(node, ast.Num): return node.n
        if isinstance(node, ast.BinOp):
            return self.operators[type(node.op)](self.evaluate_node(node.left), self.evaluate_node(node.right))
        if isinstance(node, ast.UnaryOp):
            return self.operators[type(node.op)](self.evaluate_node(node.operand))
        raise TypeError(f"Unsupported node: {type(node)}")

    def compute(self, raw_input: str) -> float:
        equation = self.clean_scratch(raw_input)
        if not equation: return 0.0
        try:
            tree = ast.parse(equation, mode='eval').body
            result = self.evaluate_node(tree)
            logging.info(f"Algebra: {equation} = {result}")
            return result
        except Exception as e:
            logging.error(f"Parse failure: {e}")
            return 0.0