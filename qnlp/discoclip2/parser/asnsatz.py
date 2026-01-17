from typing import Mapping

from lambeq.ansatz.tensor import SplitTensorAnsatz, Dim
from lambeq.backend.grammar import Ty, Word, Cup
from lambeq.backend import grammar

class CustomMPSAnsatz(SplitTensorAnsatz):
    """Split large boxes into matrix product states."""

    BOND_TYPE: Ty = Ty('B')

    def __init__(self,
                 ob_map: Mapping[Ty, Dim],
                 bond_dim: int,
                 max_order: int = 3,
                 uncurry_left: bool = True) -> None:
        """Instantiate a matrix product state ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`lambeq.backend.grammar.Ty` to the
            dimension space it uses in a tensor network.
        bond_dim: int
            The size of the bonding dimension.
        max_order: int
            The maximum order of each tensor in the matrix product
            state, which must be at least 3.
        uncurry_left: bool
            If True, the uncurrying cups are placed on the left-hand
            side. If False, they are placed on the right-hand side.

        """
        if max_order < 3:
            raise ValueError('`max_order` must be at least 3')
        if self.BOND_TYPE in ob_map:
            raise ValueError('specify bond dimension using `bond_dim`')
        ob_map = dict(ob_map)
        ob_map[self.BOND_TYPE] = Dim(bond_dim)

        super().__init__(ob_map, uncurry_left)

        self.bond_dim = bond_dim
        self.max_order = max_order
        self.split_functor = grammar.Functor(
            grammar.grammar,
            ob=lambda _, ob: ob,
            ar=self._split_ar
        )

    def _split_ar(self, _: grammar.Functor,
                  ar: grammar.Box) -> grammar.Diagrammable:
        if len(ar.dom) + len(ar.cod) <= 1:
            return grammar.Box(f'{ar.name}_0', ar.dom, ar.cod, z=ar.z)

        if self.uncurry.matches(ar):
            return self.split_functor(self.uncurry.rewrite(ar))

        bond = self.BOND_TYPE
        boxes = []
        cups = []
        step_size = self.max_order - 2
        for i, start in enumerate(range(0, len(ar.cod), step_size)):
            cod = bond.r @ ar.cod[start:start+step_size] @ bond
            boxes.append(Word(f'{ar.name}_{i}', cod))
            cups += [grammar.Id(cod[1:-1]), Cup(bond, bond.r)]
        boxes[0] = Word(boxes[0].name, boxes[0].cod[1:])
        boxes[-1] = Word(boxes[-1].name, boxes[-1].cod[:-1])

        return (grammar.Id().tensor(*boxes)
                >> grammar.Id().tensor(*cups[:-1]))  # type: ignore[arg-type]
