# Minimal parser-free AST for a tiny Datalog fragment.
# Students can build facts/rules directly as Python objects.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable, Set, Union

class Term:
    """Base class for terms."""
    def is_var(self) -> bool: raise NotImplementedError
    def is_const(self) -> bool: raise NotImplementedError

@dataclass(frozen=True)
class Var(Term):
    name: str
    def is_var(self) -> bool: return True
    def is_const(self) -> bool: return False
    def __str__(self) -> str: return self.name

@dataclass(frozen=True)
class Const(Term):
    value: str
    def is_var(self) -> bool: return False
    def is_const(self) -> bool: return True
    def __str__(self) -> str: return self.value



# ------------------ Atoms ------------------

@dataclass(frozen=True)
class Atom:
    rel: str
    terms: Tuple[Term, ...]
    def __post_init__(self):
        if not self.rel:
            raise ValueError("Relation symbol must be non-empty.")
        if not isinstance(self.terms, tuple):
            raise TypeError("terms must be a tuple of Term.")
        if not all(isinstance(t, Term) for t in self.terms):
            raise TypeError("All terms must be Term instances.")
    @property
    def arity(self) -> int: return len(self.terms)
    @property
    def vars(self) -> Set[Var]: return {t for t in self.terms if isinstance(t, Var)}
    @property
    def consts(self) -> Set[Const]: return {t for t in self.terms if isinstance(t, Const)}
    def __str__(self) -> str:
        args = ", ".join(map(str, self.terms))
        return f"{self.rel}({args})"

# Convenience: atom("R", "a", var("X")) will coerce str -> Const
TermLike = Union[Term, str]
def atom(rel: str, *args: TermLike) -> Atom:
    terms: Tuple[Term, ...] = tuple(arg if isinstance(arg, Term) else Const(str(arg)) for arg in args)
    return Atom(rel, terms)

# ------------------ Rules ------------------

@dataclass(frozen=True)
class Rule:
    head: Atom
    body: Tuple[Atom, ...]  # positive literals only (no negation in this scaffold)
    def __post_init__(self):
        if not isinstance(self.body, list):
            raise TypeError("body must be a list of Atoms.")
        if not all(isinstance(a, Atom) for a in self.body):
            raise TypeError("All body literals must be Atom instances.")
        # Simple arity sanity: same predicate name may appear anywhere; no restriction here.
    @property
    def vars(self) -> Set[Var]:
        v = set(self.head.vars)
        for a in self.body: v |= a.vars
        return v
    @property
    def predicates(self) -> Set[str]:
        return {self.head.rel, *[a.rel for a in self.body]}
    def __str__(self) -> str:
        body_str = ", ".join(map(str, self.body))
        return f"{self.head} :- {body_str}."


