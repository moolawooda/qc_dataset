"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.


"""

import math

import numpy as np


class EngUnit:
    """Provides energy values with units and allows mathematical operations such
    as addition, subtraction, multiplication, and division, as well as a series
    of logical operations, similar to regular data types (int, float). Currently
    supports four energy units (Hartree, eV, kJ/mol, kcal/mol) which can be
    converted with high precision. The default unit is Hartree. Can be used as
    a numpy array data type for normal operations, and numpy arrays can also be
    used as energy values. The functionality supported by both will differ
    slightly, with the latter being recommended.

    Note: Logical operations cannot be performed when using numpy arrays, only
    mathematical operations are supported.

    Supported mathematical operations (the result will automatically be converted
    to Hartree when two different units are involved in the operation):
        +, -, *, /, //, %, **, abs, round, ceil, floor, trunc \n
    Supported logical operations:
        ==, !=, <, <=, >, >= \n
    Supported other functions:
        print, copy

    Internal Composition
    ----------
    Instance methods:
        is_close_to, unit_convert \n
    Class methods:
        from_ndarray, from_dataframe \n
    Static methods:
        unit_convert_factor, unit_handler \n
    Magic methods:
        __add__, __sub__, __mul__, __truediv__, __floordiv__, __mod__, __pow__,
        __radd__, __rsub__, __rmul__, __rtruediv__, __rfloordiv__, __rmod__,
        __iadd__, __isub__, __imul__, __itruediv__, __ifloordiv__, __imod__,
        __ipow__, __neg__, __abs__, __round__, __ceil__, __floor__, __trunc__,
        __eq__, __ne__, __lt__, __le__, __gt__, __ge__, __str__, __copy__ \n
    """

    def __init__(
        self, energy: int | float | np.ndarray, unit: str = "Hartree", order: int = 1
    ):
        if not isinstance(energy, (int, float, np.ndarray)):
            raise TypeError(
                "The energy value must be a int, float or numpy.ndarray, "
                f"but got {type(energy)} instead"
            )
        self.energy = energy
        self.unit = self.unit_handler(unit)
        self.order = order if self.unit != "1" else 0

    def __add__(self, other):
        if not isinstance(other, EngUnit):
            raise TypeError(
                "The object to be added must be an EngUnit object: "
                f"one is {type(self)}, the other is {type(other)}"
            )
        if not self.order == other.order:
            raise ValueError(
                "The two objects must have the same order of unit: "
                f"one is {self.order}, the other is {other.order}"
            )
        obj1 = self.unit_convert("Hartree")
        obj2 = other.unit_convert("Hartree")
        return EngUnit(obj1.energy + obj2.energy, "Hartree", self.order)

    def __sub__(self, other):
        if not isinstance(other, EngUnit):
            raise TypeError(
                "The object to be added must be an EngUnit object: "
                f"one is {type(self)}, the other is {type(other)}"
            )
        if not self.order == other.order:
            raise ValueError(
                "The two objects must have the same order of unit: "
                f"one is {self.order}, the other is {other.order}"
            )
        obj1 = self.unit_convert("Hartree")
        obj2 = other.unit_convert("Hartree")
        return EngUnit(obj1.energy - obj2.energy, "Hartree", self.order)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return EngUnit(self.energy * other, self.unit, self.order)
        elif isinstance(other, EngUnit):
            obj1 = self.unit_convert("Hartree")
            obj2 = other.unit_convert("Hartree")
            return EngUnit(
                obj1.energy * obj2.energy, "Hartree", obj1.order + obj2.order
            )
        else:
            raise TypeError(
                "Invalid data type for multiplication: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return EngUnit(self.energy / other, self.unit, self.order)
        elif isinstance(other, EngUnit):
            obj1 = self.unit_convert("Hartree")
            obj2 = other.unit_convert("Hartree")
            return EngUnit(
                obj1.energy / obj2.energy, "Hartree", obj1.order - obj2.order
            )
        else:
            raise TypeError(
                "Invalid data type for division: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __floordiv__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return EngUnit(self.energy // other, self.unit, self.order)
        elif isinstance(other, EngUnit):
            obj1 = self.unit_convert("Hartree")
            obj2 = other.unit_convert("Hartree")
            return EngUnit(
                obj1.energy // obj2.energy, "Hartree", obj1.order - obj2.order
            )
        else:
            raise TypeError(
                "Invalid data type for floor division: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __mod__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return EngUnit(self.energy % other, self.unit, self.order)
        elif isinstance(other, EngUnit):
            obj1 = self.unit_convert("Hartree")
            obj2 = other.unit_convert("Hartree")
            return EngUnit(
                obj1.energy % obj2.energy, "Hartree", obj1.order - obj2.order
            )
        else:
            raise TypeError(
                "Invalid data type for mod operation: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __pow__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return EngUnit(self.energy**other, self.unit, self.order * other)
        else:
            raise TypeError(
                "Invalid data type for power operation: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return EngUnit(other / self.energy, self.unit, -self.order)
        elif isinstance(other, EngUnit):
            return other.__truediv__(self)

    def __rfloordiv__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return EngUnit(other // self.energy, self.unit, -self.order)
        elif isinstance(other, EngUnit):
            return other.__floordiv__(self)

    def __rmod__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return EngUnit(other % self.energy, self.unit, -self.order)
        elif isinstance(other, EngUnit):
            return other.__mod__(self)

    def __iadd__(self, other):
        if not isinstance(other, EngUnit):
            raise TypeError(
                "The object to be added must be an EngUnit object: "
                f"one is {type(self)}, the other is {type(other)}"
            )
        if not self.order == other.order:
            raise ValueError(
                "The two objects must have the same order of unit: "
                f"one is {self.order}, the other is {other.order}"
            )
        self.energy += other.unit_convert("Hartree", copy=False).energy
        return self

    def __isub__(self, other):
        if not isinstance(other, EngUnit):
            raise TypeError(
                "The object to be added must be an EngUnit object: "
                f"one is {type(self)}, the other is {type(other)}"
            )
        if not self.order == other.order:
            raise ValueError(
                "The two objects must have the same order of unit: "
                f"one is {self.order}, the other is {other.order}"
            )
        self.energy -= other.unit_convert("Hartree", copy=False).energy
        return self

    def __imul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            self.energy *= other
            return self
        elif isinstance(other, EngUnit):
            self.energy *= other.unit_convert("Hartree", copy=False).energy
            self.order += other.order
            return self
        else:
            raise TypeError(
                "Invalid data type for multiplication: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __itruediv__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            self.energy /= other
            return self
        elif isinstance(other, EngUnit):
            self.energy /= other.unit_convert("Hartree", copy=False).energy
            self.order -= other.order
            return self
        else:
            raise TypeError(
                "Invalid data type for division: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __ifloordiv__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            self.energy //= other
            return self
        elif isinstance(other, EngUnit):
            self.energy //= other.unit_convert("Hartree", copy=False).energy
            self.order -= other.order
            return self
        else:
            raise TypeError(
                "Invalid data type for floor division: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __imod__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            self.energy %= other
            return self
        elif isinstance(other, EngUnit):
            self.energy %= other.unit_convert("Hartree", copy=False).energy
            self.order -= other.order
            return self
        else:
            raise TypeError(
                "Invalid data type for mod operation: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __ipow__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            self.energy **= other
            self.order *= other
            return self
        else:
            raise TypeError(
                "Invalid data type for power operation: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __neg__(self):
        return EngUnit(-self.energy, self.unit, self.order)

    def __abs__(self):
        return EngUnit(abs(self.energy), self.unit, self.order)

    def __round__(self, n: int = 0):
        if isinstance(self.energy, np.ndarray):
            return EngUnit(np.round(self.energy, n), self.unit, self.order)
        return EngUnit(round(self.energy, n), self.unit, self.order)

    def __ceil__(self):
        return EngUnit(math.ceil(self.energy), self.unit, self.order)

    def __floor__(self):
        return EngUnit(math.floor(self.energy), self.unit, self.order)

    def __trunc__(self):
        return EngUnit(math.trunc(self.energy), self.unit, self.order)

    def __eq__(self, other) -> bool:
        if isinstance(other, EngUnit):
            obj1 = self.unit_convert("Hartree")
            obj2 = other.unit_convert("Hartree")
            return (obj1.energy == obj2.energy) and (obj1.order == obj2.order)
        elif isinstance(other, (int, float)) and (self.order == 0):
            return self.energy == other
        else:
            raise TypeError(
                "Invalid data type for comparison: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other) -> bool:
        if isinstance(self.energy, np.ndarray):
            raise TypeError("Comparison is not supported for numpy array")
        if isinstance(other, EngUnit):
            if self.order != other.order:
                raise ValueError("Two objects must have the same order of unit")
            if isinstance(other.energy, np.ndarray):
                raise TypeError("Comparison is not supported for numpy array")
            obj1 = self.unit_convert("Hartree")
            obj2 = other.unit_convert("Hartree")
            return obj1.energy < obj2.energy
        elif isinstance(other, (int, float)) and (self.order == 0):
            return self.energy < other
        else:
            raise TypeError(
                "Invalid data type for comparison: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __le__(self, other) -> bool:
        if isinstance(self.energy, np.ndarray):
            raise TypeError("Comparison is not supported for numpy array")
        if isinstance(other, EngUnit):
            if self.order != other.order:
                raise ValueError("Two objects must have the same order of unit")
            if isinstance(other.energy, np.ndarray):
                raise TypeError("Comparison is not supported for numpy array")
            obj1 = self.unit_convert("Hartree")
            obj2 = other.unit_convert("Hartree")
            return obj1.energy <= obj2.energy
        elif isinstance(other, (int, float)) and (self.order == 0):
            return self.energy <= other
        else:
            raise TypeError(
                "Invalid data type for comparison: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    def __gt__(self, other) -> bool:
        return not self.__le__(other)

    def __ge__(self, other) -> bool:
        return not self.__lt__(other)

    def __str__(self) -> str:
        if self.order == 1:
            return f"{self.energy} {self.unit}"
        elif self.order == 0:
            return f"{self.energy}"
        else:
            return f"{self.energy} ({self.unit})^{self.order}"

    def __copy__(self):
        return EngUnit(self.energy, self.unit, self.order)

    # TODO test this for numpy array
    def is_close_to(self, other, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
        if isinstance(other, EngUnit):
            obj1 = self.unit_convert("Hartree")
            obj2 = other.unit_convert("Hartree")
            return math.isclose(
                obj1.energy, obj2.energy, rel_tol=rel_tol, abs_tol=abs_tol
            ) and (self.order == other.order)
        elif isinstance(other, (int, float)) and (self.order == 0):
            return math.isclose(self.energy, other, rel_tol=rel_tol, abs_tol=abs_tol)
        else:
            raise TypeError(
                "Invalid data type for comparison: "
                f"one is {type(self)}, the other is {type(other)}"
            )

    @staticmethod
    def unit_handler(unit: str):
        # correct the unit to the standard form
        unit_mapping = {
            "ev": "eV",
            "electronvolt": "eV",
            "electron volt": "eV",
            "kcal/mol": "kcal/mol",
            "kcal": "kcal/mol",
            "kilocalorie/mol": "kcal/mol",
            "kilocalorie": "kcal/mol",
            "kilocalorie per mole": "kcal/mol",
            "kcal per mol": "kcal/mol",
            "kj/mol": "kJ/mol",
            "kj": "kJ/mol",
            "kilojoule/mol": "kJ/mol",
            "kilojoule": "kJ/mol",
            "kilojoule per mole": "kJ/mol",
            "kj per mol": "kJ/mol",
            "hartree": "Hartree",
            "ha": "Hartree",
            "au": "Hartree",
            "a.u.": "Hartree",
            "atomic unit": "Hartree",
            "atomic units": "Hartree",
        }
        unit_lower = unit.lower()
        if unit_lower in unit_mapping:
            return unit_mapping[unit_lower]
        else:
            raise ValueError("Invalid unit")

    def unit_convert(self, target_unit: str, copy: bool = True):
        target_unit = self.unit_handler(target_unit)

        if self.unit == target_unit:
            return self.__copy__()

        factor = self.unit_convert_factor(self.unit, target_unit)
        if copy:
            # return a new object with the original object unchanged
            obj = EngUnit(self.energy * (factor**self.order), target_unit, self.order)
            return obj
        else:
            # change the unit of the original object in place
            self.energy = self.energy * (factor**self.order)
            self.unit = target_unit

    @staticmethod
    def unit_convert_factor(old_unit: str, new_unit: str) -> int | float:
        # factor copied from Wolfram Alpha
        # kcal used here is thermochemical kcal
        factors: dict[str, dict[str, int | float]] = {
            "Hartree": {
                "Hartree": 1,
                "eV": 27.211386246,
                "kJ/mol": 2625.49963948,
                "kcal/mol": 627.50947406,
            },
            "eV": {
                "Hartree": 0.036749322176,
                "eV": 1,
                "kJ/mol": 96.4853321233100184,
                "kcal/mol": 23.060547830619029254302103250478,
            },
            "kJ/mol": {
                "Hartree": 0.00038087988471,
                "eV": 0.010364269656262173798435493528601,
                "kJ/mol": 1,
                "kcal/mol": 0.23900573613766730401529636711281,
            },
            "kcal/mol": {
                "Hartree": 0.00159360143764,
                "eV": 0.043364104241800935172654104923667,
                "kJ/mol": 4.184,
                "kcal/mol": 1,
            },
        }
        return factors[old_unit][new_unit]


if __name__ == "__main__":
    a = EngUnit(1, "Hartree")
    b = EngUnit(1, "eV")
    c = EngUnit(1, "kJ/mol")
    d = EngUnit(1, "kcal/mol")

    e = a + b
    print(e)  # 1.036749322176 Hartree
    f = e.unit_convert("kcal/mol")
    print(f)  # 650.5700218907233 kcal/mol

    # g = np.array([a, b, c, d])
    g = EngUnit(np.array([1, 2, 3, 4]), "Hartree")
    h = 5 * g / 2 + g * 3
    y = a * h

    print(g)
    print(h)
    print(y)

    g.unit_convert("kcal/mol", copy=False)
    print(g)

    # org_array = np.array([1, 2, 3, 4])
    # eng_array = EngUnit.from_ndarray(org_array, "eV")
    # print(org_array)
    # for item in eng_array:
    #     print(item)
