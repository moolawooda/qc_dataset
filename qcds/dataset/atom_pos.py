"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


class AtomPosition:
    """one atom's position in a molecule."""

    DEFAULT_UNIT = "angstrom"

    def __init__(
        self, atom: str, x: float, y: float, z: float, unit: str = DEFAULT_UNIT
    ) -> None:
        """Create an AtomPosition object.

        Args:
            atom (str): element symbol
            x (float): x-coordinate
            y (float): y-coordinate
            z (float): z-coordinate
            unit (str, optional): unit of the coordinates. Defaults to "angstrom".

        Raises:
            TypeError: _description_
            ValueError: _description_
        """
        if not isinstance(atom, str):
            raise TypeError("Atom must be a string")
        if not isinstance(x, (int, float)):
            try:
                x = float(x)
            except ValueError:
                raise ValueError("x must be a number")
        if not isinstance(y, (int, float)):
            try:
                y = float(y)
            except ValueError:
                raise ValueError("y must be a number")
        if not isinstance(z, (int, float)):
            try:
                z = float(z)
            except ValueError:
                raise ValueError("z must be a number")
        if not isinstance(unit, str):
            raise TypeError("Unit must be a string")

        self.atom = atom
        self.x = x
        self.y = y
        self.z = z
        self.unit = unit

    def __repr__(self) -> str:
        return f"AtomPosition({self.atom}, {self.x:f} {self.y:f} {self.z:f}, {self.unit})"

    def __str__(self) -> str:
        return f"{self.atom}  {self.x:f} {self.y:f} {self.z:f}"

    def __hash__(self) -> int:
        return hash((self.atom, self.x, self.y, self.z, self.unit))

    def unit_convert(self, target_unit: str = DEFAULT_UNIT, copy: bool = False):
        pass  # TODO


if __name__ == "__main__":
    atom = AtomPosition("C", 0.0, 0.0, 0.000005)
    print(atom)
    print(repr(atom))
