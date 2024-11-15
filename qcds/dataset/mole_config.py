"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from qcds.dataset.atom_pos import AtomPosition


class MoleConfig:
    """the structure of a molecule"""

    DEFAULT_SPIN_TYPE = "gaussian"

    def __init__(
        self,
        name: str,
        pos_list: list[AtomPosition],
        charge: int,
        spin: int,
        spin_type: str = DEFAULT_SPIN_TYPE,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(pos_list, list):
            raise TypeError("pos_list must be a list")
        if not all(isinstance(i, AtomPosition) for i in pos_list):
            raise TypeError("pos_list must contain only AtomPosition objects")
        if not isinstance(charge, int):
            try:
                charge = int(charge)
            except ValueError:
                raise ValueError("charge must be an integer")
        if not isinstance(spin, int):
            try:
                spin = int(spin)
            except ValueError:
                raise ValueError("spin must be an integer")
        if not isinstance(spin_type, str):
            raise TypeError("spin_type must be a string")
        if spin_type.lower() not in ["pyscf", "gaussian"]:
            raise ValueError("Invalid spin_type, must be 'pyscf' or 'gaussian'")

        self.name = name
        self.pos_list = pos_list
        self.charge = charge

        # spin was stored in gaussian type
        match spin_type.lower():
            case "pyscf":
                self.spin = spin + 1
            case "gaussian":
                self.spin = spin
            case _:
                raise ValueError("Invalid spin_type, must be 'pyscf' or 'gaussian'")

        self.atom_dict: dict = {}
        self.get_atom_dict()

        self.xyz_str = self.get_xyz_str()

    @classmethod
    def from_xyz(cls, xyz_filename: str):
        with open(xyz_filename, "r") as f:
            lines = f.readlines()
        name = xyz_filename[:-4]
        num_atoms = int(lines[0].strip())
        charge, spin = map(int, lines[1].strip().split())
        pos_list = []
        for line in lines[2 : 2 + num_atoms + 1]:
            atom, x, y, z = line.strip().split()
            pos_list.append(AtomPosition(atom, float(x), float(y), float(z)))
        return cls(name, pos_list, charge, spin)

    def get_atom_dict(self):
        for i in self.pos_list:
            if i.atom not in self.atom_dict:
                self.atom_dict[i.atom] = [[i.x, i.y, i.z]]
            else:
                self.atom_dict[i.atom] += [[i.x, i.y, i.z]]
        return self

    def get_xyz_str(self) -> str:
        str_list = [f"{i}" for i in self.pos_list]
        return "\n".join(str_list)

    def get_spin(self, spin_type: str = DEFAULT_SPIN_TYPE) -> int:
        if not isinstance(spin_type, str):
            raise TypeError("spin_type must be a string")

        match spin_type.lower():
            case "pyscf":
                result = self.spin - 1
            case "gaussian":
                result = self.spin
            case _:
                raise ValueError("Invalid spin_type, must be 'pyscf' or 'gaussian'")
        return result

    def __str__(self) -> str:
        return f"{self.charge} {self.spin}\n{self.xyz_str}"

    def __repr__(self) -> str:
        return f"MoleculeConfiguration(name={self.name}, geom=\n{self.__str__()}\n)"

    def __hash__(self) -> int:
        return hash((self.name, self.charge, self.spin, tuple(self.pos_list)))


if __name__ == "__main__":
    atom1 = AtomPosition("C", 0.0, 0.0, 0.0)
    atom2 = AtomPosition("H", 0.0, 0.0, 1.0)
    geom = MoleConfig("test_mol", [atom1, atom2], 0, 1)
    print(geom)
    print(repr(geom))

    geom2 = MoleConfig("test_mol2", [], 0, 1)
