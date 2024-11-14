"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import csv
import os
from string import Template

import pkg_resources as pkg  # type: ignore

from qcds.dataset.mole_config import MoleConfig
from qcds.tools.elem_mass import elem_mass
from qcds.tools.eng_unit import EngUnit


class SubSet:
    def __init__(
        self,
        name: str,
        dataset_eval_file: str,
        geom_path: str,
        list_file: str | None = None,
    ):
        self.name = name

        self.dataset_eval_file = dataset_eval_file
        self.geom_path = geom_path
        self.list_file = list_file

        self.mole_configs: list[MoleConfig] = []
        # self.mole_paths: dict = {}
        self.input_path: str = ""

        self.dataset_eval: list[dict] = []
        self.mole_eng: dict = {}

    @staticmethod
    def read_moleconfigs(
        geom_path: str, list_file: str | None = None
    ) -> list[MoleConfig]:
        """read *.xyz files in the geom_path. The use of MoleConfig() instead
        of string is to make the geometry manipulation easier for different
        softwares, i.e., abacus, pyscf, and gaussian etc.

        Args:
            geom_path (str): path to the geometry files
        """
        if list_file is None:
            return [
                MoleConfig.from_xyz(f)
                for f in os.listdir(geom_path)
                if f.endswith(".xyz")
            ]
        else:
            with open(list_file, "r") as f:
                moles = f.readlines()
            return [
                MoleConfig.from_xyz(f)
                for f in moles
                if f.endswith(".xyz") and os.path.exists(os.path.join(geom_path, f))
            ]

    @staticmethod
    def read_dataseteval(csv_file: str, eng_unit: str = "kcal/mol") -> list[dict]:
        """read the dataseteval file. The file must be in csv format with following
        columns:
            name_of_this_item, (stoichiometry_number, mole_name)*n, eng_ref

        Args:
            csv_file (str): the dataseteval file
            eng_unit (str, optional): energy unit. Defaults to "kcal/mol".

        Returns:
            list: _description_
        """
        with open(csv_file, "r", newline="") as f:
            reader = csv.reader(f)

        eval_list: list = []

        for row in reader:
            name = row[0]
            eng_ref = EngUnit(float(row[-1]), unit=eng_unit)
            moles = row[2:-1:2]
            stoichs = list(map(int, row[1:-1:2]))
            eval_list.append(
                {"name": name, "eng_ref": eng_ref, "moles": moles, "stoichs": stoichs}
            )

        return eval_list

    def input_gen_abacus(
        self, params_input: dict, params_stru: dict, params_kpt: dict, params_sh: dict
    ):
        def get_pot(element: str, pot_path: str) -> str:
            # TODO
            pot_file = f"PROJECT-{element}_HGH_NLCC.UPF-1.upf"
            if not os.path.exists(os.path.join(pot_path, pot_file)):
                raise FileNotFoundError(f"Potential file {pot_file} not found")
            return pot_file

        print("Generating input files for Abacus...")

        input_file = pkg.resource_string("qcds", "templates/abacus_input.template")
        input_template = Template(input_file.decode("utf-8"))
        stru_file = pkg.resource_string("qcds", "templates/abacus_stru.template")
        stru_template = Template(stru_file.decode("utf-8"))
        kpt_file = pkg.resource_string("qcds", "templates/abacus_kpt.template")
        kpt_template = Template(kpt_file.decode("utf-8"))
        sh_file = pkg.resource_string("qcds", "templates/sh.template")
        sh_template = Template(sh_file.decode("utf-8"))

        for mole in self.mole_configs:
            new_folder = os.path.join(self.input_path, mole.name)
            # self.mole_paths.update({mole.name: new_folder})
            print(f"> {mole.name}")

            if not os.path.exists(new_folder):
                os.makedirs(new_folder)

            spin = mole.get_spin("pyscf")
            params_input.update(
                {
                    "nspin": 1 if (spin == 0) else 2,
                    "nupdown": spin,
                    "nupdown_tag": "#nupdown" if (spin == 0) else "nupdown ",
                }
            )

            potential_str = ""
            xyz_str = ""
            for atom, coord in mole.get_atom_dict().items():
                xyz_str += f"{atom}\n0\n{len(coord)}\n{
                    ''.join(
                        [
                            f'{x[0]:.8f} {x[1]:.8f} {x[2]:.8f}\n' for x in coord
                        ]
                    )
                }"
                pot = get_pot(atom, params_input["pseudo_dir"])
                potential_str += f"{atom} {elem_mass[atom]} {pot}\n"

            params_stru.update(
                {
                    "XYZS": xyz_str,
                    "ATOMIC_SPECIES": potential_str,
                }
            )

            params_sh.update(
                {
                    "jobname": mole.name,
                    "commands": (
                        "export OMP_NUM_THREADS=1\n"
                        f"mpirun -np {params_sh["cores_per_task"]} abacus"
                    ),
                }
            )

            with open(os.path.join(new_folder, "INPUT"), "w") as f:
                f.write(input_template.substitute(params_input))
            with open(os.path.join(new_folder, "STRU"), "w") as f:
                f.write(stru_template.substitute(params_stru))
            with open(os.path.join(new_folder, "KPT"), "w") as f:
                f.write(kpt_template.substitute(params_kpt))
            with open(os.path.join(new_folder, f"{mole.name}.sh"), "w") as f:
                f.write(sh_template.substitute(params_sh))

        print("Done!")

    def input_gen_pyscf(self, params_pyscf: dict, params_sh: dict):
        print("Generating input files for PySCF...")

        input_file = pkg.resource_string("qcds", "templates/pyscf.template")
        input_template = Template(input_file.decode("utf-8"))
        sh_file = pkg.resource_string("qcds", "templates/sh.template")
        sh_template = Template(sh_file.decode("utf-8"))

        for mole in self.mole_configs:
            new_folder = self.input_path
            # self.mole_paths.update({mole.name: new_folder})
            print(f"> {mole.name}")

            params_pyscf.update(
                {
                    "charge": mole.charge,
                    "spin": mole.get_spin("pyscf"),
                    "atom": mole.get_xyz_str(),
                }
            )

            params_sh.update(
                {
                    "jobname": mole.name,
                    "commands": f"python {mole.name}.py > {mole.name}.log",
                }
            )

            with open(os.path.join(new_folder, f"{mole.name}.py"), "w") as f:
                f.write(input_template.substitute(params_pyscf))
            with open(os.path.join(new_folder, f"{mole.name}.sh"), "w") as f:
                f.write(sh_template.substitute(params_sh))

        print("Done!")

    def input_gen_gaussian(self, params_gaussian: dict, params_sh: dict):
        print("Generating input files for Gaussian...")

        input_file = pkg.resource_string("qcds", "templates/gaussian.template")
        input_template = Template(input_file.decode("utf-8"))
        sh_file = pkg.resource_string("qcds", "templates/sh.template")
        sh_template = Template(sh_file.decode("utf-8"))

        for mole in self.mole_configs:
            new_folder = self.input_path
            # self.mole_paths.update({mole.name: new_folder})
            print(f"> {mole.name}")

            params_gaussian.update({"chk": f"{mole.name}.chk", "xyz": str(mole)})

            params_sh.update(
                {
                    "jobname": mole.name,
                    "commands": f"g16 {mole.name}.gjf",
                }
            )

            with open(os.path.join(new_folder, f"{mole.name}.gjf"), "w") as f:
                f.write(input_template.substitute(params_gaussian))
            with open(os.path.join(new_folder, f"{mole.name}.sh"), "w") as f:
                f.write(sh_template.substitute(params_sh))

        print("Done!")

    def input_gen_psi4(self, params_psi4: dict, params_sh: dict):
        print("Generating input files for Psi4...")

        input_file = pkg.resource_string("qcds", "templates/psi4.template")
        input_template = Template(input_file.decode("utf-8"))
        sh_file = pkg.resource_string("qcds", "templates/sh.template")
        sh_template = Template(sh_file.decode("utf-8"))

        for mole in self.mole_configs:
            new_folder = self.input_path
            # self.mole_paths.update({mole.name: new_folder})
            print(f"> {mole.name}")

            params_psi4.update(
                {
                    "xyz": str(mole),
                    "reference": "uks" if (mole.get_spin() > 1) else "rks",
                }
            )

            params_sh.update(
                {
                    "jobname": mole.name,
                    "commands": f"python {mole.name}.py > {mole.name}.log",
                }
            )

            with open(os.path.join(new_folder, f"{mole.name}.py"), "w") as f:
                f.write(input_template.substitute(params_psi4))
            with open(os.path.join(new_folder, f"{mole.name}.sh"), "w") as f:
                f.write(sh_template.substitute(params_sh))

        print("Done")

    def input_gen(
        self, software: str, path: str, prefix: str, suffix: str, params: dict
    ):
        self.mole_configs = self.read_moleconfigs(self.geom_path, self.list_file)

        self.input_path = os.path.join(
            path, f"{prefix}_{software}_{self.name}_{suffix}"
        )
        if not os.path.exists(self.input_path):
            os.makedirs(self.input_path)

        match software.lower():
            case "abacus":
                self.input_gen_abacus(
                    params["abacus_input"],
                    params["abacus_stru"],
                    params["abacus_kpt"],
                    params["sh"],
                )
            case "pyscf":
                self.input_gen_pyscf(params["pyscf"], params["sh"])
            case "gaussian":
                self.input_gen_gaussian(params["gaussian"], params["sh"])
            case "psi4":
                self.input_gen_psi4(params["psi4"], params["sh"])
            case _:
                raise ValueError("Software not supported")

    def output_read_abacus(self):
        pass

    def output_read_pyscf(self):
        pass

    def output_read_gaussian(self):
        pass

    def output_read_psi4(self):
        pass

    def output_read(self, software: str):
        match software.lower():
            case "abacus":
                self.output_read_abacus()
            case "pyscf":
                self.output_read_pyscf()
            case "gaussian":
                self.output_read_gaussian()
            case "psi4":
                self.output_read_psi4()
            case _:
                raise ValueError("Software not supported")

    def eval(self, target_unit: str = "kcal/mol", output_file: str = "eval_result.csv"):
        self.dataset_eval = self.read_dataseteval(self.dataset_eval_file)

        eval_result: list[list] = [["name", "eng_ref", "eng_calc", "eng_err"]]
        for item in self.dataset_eval:
            name = item["name"]
            eng_ref: EngUnit = item["eng_ref"]

            eng_calc = EngUnit(0)
            for mole, stoich in zip(item["moles"], item["stoichs"]):
                eng_calc += self.mole_eng[mole] * stoich

            eng_err: EngUnit = eng_calc - eng_ref

            eng_ref.unit_convert(target_unit, copy=False)
            eng_calc.unit_convert(target_unit, copy=False)
            eng_err.unit_convert(target_unit, copy=False)

            eval_result.append([name, eng_ref.energy, eng_calc.energy, eng_err.energy])

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(eval_result)
