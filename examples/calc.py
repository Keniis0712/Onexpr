"""Tiny calculator: tokenize, parse, evaluate with match/case."""

import re
from dataclasses import dataclass


@dataclass
class Num:
    value: float


@dataclass
class Op:
    op: str  # '+', '-', '*', '/'


@dataclass
class LParen: pass


@dataclass
class RParen: pass


def tokenize(src):
    pos = 0
    while pos < len(src):
        ch = src[pos]
        if ch.isspace():
            pos += 1
            continue
        if ch.isdigit() or ch == ".":
            m = re.match(r"\d+(?:\.\d+)?", src[pos:])
            yield Num(float(m.group(0)))
            pos += m.end()
            continue
        if ch in "+-*/":
            yield Op(ch)
            pos += 1
            continue
        if ch == "(":
            yield LParen()
            pos += 1
            continue
        if ch == ")":
            yield RParen()
            pos += 1
            continue
        raise ValueError(f"unexpected character: {ch!r}")


class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def eat(self):
        t = self.tokens[self.i]
        self.i += 1
        return t

    def parse_expr(self):
        left = self.parse_term()
        while True:
            match self.peek():
                case Op(op="+") | Op(op="-"):
                    op = self.eat().op
                    right = self.parse_term()
                    left = ("bin", op, left, right)
                case _:
                    return left

    def parse_term(self):
        left = self.parse_factor()
        while True:
            match self.peek():
                case Op(op="*") | Op(op="/"):
                    op = self.eat().op
                    right = self.parse_factor()
                    left = ("bin", op, left, right)
                case _:
                    return left

    def parse_factor(self):
        match self.eat():
            case Num(value=v):
                return ("num", v)
            case LParen():
                inner = self.parse_expr()
                if not isinstance(self.eat(), RParen):
                    raise ValueError("expected )")
                return inner
            case other:
                raise ValueError(f"unexpected token: {other}")


def evaluate(node):
    match node:
        case ("num", v):
            return v
        case ("bin", "+", a, b):
            return evaluate(a) + evaluate(b)
        case ("bin", "-", a, b):
            return evaluate(a) - evaluate(b)
        case ("bin", "*", a, b):
            return evaluate(a) * evaluate(b)
        case ("bin", "/", a, b):
            return evaluate(a) / evaluate(b)


def calc(src):
    tokens = tokenize(src)
    tree = Parser(tokens).parse_expr()
    return evaluate(tree)


for src in ["1 + 2", "2 * 3 + 4", "(1 + 2) * 3", "10 / 4 - 1"]:
    print(f"{src} = {calc(src)}")
