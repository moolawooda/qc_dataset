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
        self.out_path: str = ""

        self.dataset_eval: list[dict] = []
        self.mole_eng: dict = {}

        self.abacus_pseudo_file_syntax: str = ""
        self.params_abacus_input: dict = {}
        self.params_abacus_stru: dict = {}
        self.params_abacus_kpt: dict = {}
        self.params_pyscf: dict = {}
        self.params_gaussian: dict = {}
        self.params_psi4: dict = {}
        self.params_slurm: dict = {}

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
                MoleConfig.from_xyz(os.path.join(geom_path, f))
                for f in os.listdir(geom_path)
                if f.endswith(".xyz")
            ]
        else:
            with open(list_file, "r") as f:
                moles = f.readlines()
            return [
                MoleConfig.from_xyz(os.path.join(geom_path, f))
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

    def params_def_abacus(
        self,
        pseudo_dir: str,
        pseudo_file_syntax: str,
        calculation: str,
        dft_functional: str,
        ecutwfc: int,
        lattice: float | None = None,
        lattice_x: float | None = None,
        lattice_y: float | None = None,
        lattice_z: float | None = None,
    ):
        self.abacus_pseudo_file_syntax = pseudo_file_syntax

        self.params_abacus_input = {
            "pseudo_dir": pseudo_dir,
            "calculation": calculation,
            "dft_functional": dft_functional,
            "ecutwfc": ecutwfc,
        }

        match (lattice, lattice_x, lattice_y, lattice_z):
            case (None, None, None, None):
                raise ValueError("Lattice parameters are missing")
            case (None, _, _, _):
                self.params_abacus_stru = {
                    "LATTICE_X": lattice_x,
                    "LATTICE_Y": lattice_y,
                    "LATTICE_Z": lattice_z,
                }
            case (_, None, None, None):
                self.params_abacus_stru = {
                    "LATTICE_X": lattice,
                    "LATTICE_Y": lattice,
                    "LATTICE_Z": lattice,
                }
            case _:
                raise ValueError("Lattice parameters are not consistent")

    def params_def_pyscf(self, num_threads: int, basis: str, xc: str):
        self.params_pyscf = {
            "num_threads": num_threads,
            "basis": basis,
            "xc": xc,
        }

    def params_def_gaussian(self, mem: str, nproc: int, command: str):
        self.params_gaussian = {
            "mem": mem,
            "nproc": nproc,
            "command": command,
        }

    def params_def_psi4(self, num_threads: int, memory: str, basis: str, xc: str):
        self.params_psi4 = {
            "num_threads": num_threads,
            "memory": memory,
            "basis": basis,
            "xc": xc,
        }

    def params_def_slurm(
        self,
        partition: str,
        qos: str,
        nodes: int,
        task_per_node: int,
        cores_per_task: int,
        slurm_output: str = "job.%j.out",
    ):
        self.params_slurm = {
            "partition": partition,
            "qos": qos,
            "nodes": nodes,
            "task_per_node": task_per_node,
            "cores_per_task": cores_per_task,
            "slurm_output": slurm_output,
        }

    def input_gen_abacus(self):
        def get_pp(element: str, pp_path: str, pp_syntax: str) -> str:
            pp_file = pp_syntax.replace("{}", element)
            if not os.path.exists(os.path.join(pp_path, pp_file)):
                raise FileNotFoundError(f"Potential file {pp_file} not found")
            return pp_file

        print("Generating input files for Abacus...")

        input_file = pkg.resource_string("qcds", "templates/abacus_input.template")
        input_template = Template(input_file.decode("utf-8"))
        stru_file = pkg.resource_string("qcds", "templates/abacus_stru.template")
        stru_template = Template(stru_file.decode("utf-8"))
        kpt_file = pkg.resource_string("qcds", "templates/abacus_kpt.template")
        kpt_template = Template(kpt_file.decode("utf-8"))
        slurm_file = pkg.resource_string("qcds", "templates/slurm.template")
        slurm_template = Template(slurm_file.decode("utf-8"))

        for mole in self.mole_configs:
            new_folder = os.path.join(self.input_path, mole.name)
            # self.mole_paths.update({mole.name: new_folder})
            print(f"> {mole.name}")

            if not os.path.exists(new_folder):
                os.makedirs(new_folder)

            params_input = self.params_abacus_input.copy()
            params_stru = self.params_abacus_stru.copy()
            params_kpt = self.params_abacus_kpt.copy()
            params_slurm = self.params_slurm.copy()

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
                pp = get_pp(
                    atom, params_input["pseudo_dir"], self.abacus_pseudo_file_syntax
                )
                potential_str += f"{atom} {elem_mass[atom]} {pp}\n"

            params_stru.update(
                {
                    "XYZS": xyz_str,
                    "ATOMIC_SPECIES": potential_str,
                }
            )

            params_slurm.update(
                {
                    "jobname": mole.name,
                    "commands": (
                        "export OMP_NUM_THREADS=1\n"
                        f"mpirun -np {params_slurm["cores_per_task"]} abacus"
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
                f.write(slurm_template.substitute(params_slurm))

        print("Done!")

    def input_gen_pyscf(self):
        print("Generating input files for PySCF...")

        input_file = pkg.resource_string("qcds", "templates/pyscf.template")
        input_template = Template(input_file.decode("utf-8"))
        slurm_file = pkg.resource_string("qcds", "templates/slurm.template")
        slurm_template = Template(slurm_file.decode("utf-8"))

        for mole in self.mole_configs:
            new_folder = self.input_path
            # self.mole_paths.update({mole.name: new_folder})
            print(f"> {mole.name}")

            params_pyscf = self.params_pyscf.copy()
            params_slurm = self.params_slurm.copy()

            params_pyscf.update(
                {
                    "charge": mole.charge,
                    "spin": mole.get_spin("pyscf"),
                    "atom": mole.get_xyz_str(),
                }
            )

            params_slurm.update(
                {
                    "jobname": mole.name,
                    "commands": f"python {mole.name}.py > {mole.name}.log",
                }
            )

            with open(os.path.join(new_folder, f"{mole.name}.py"), "w") as f:
                f.write(input_template.substitute(params_pyscf))
            with open(os.path.join(new_folder, f"{mole.name}.sh"), "w") as f:
                f.write(slurm_template.substitute(params_slurm))

        print("Done!")

    def input_gen_gaussian(self):
        print("Generating input files for Gaussian...")

        input_file = pkg.resource_string("qcds", "templates/gaussian.template")
        input_template = Template(input_file.decode("utf-8"))
        slurm_file = pkg.resource_string("qcds", "templates/slurm.template")
        slurm_template = Template(slurm_file.decode("utf-8"))

        for mole in self.mole_configs:
            new_folder = self.input_path
            # self.mole_paths.update({mole.name: new_folder})
            print(f"> {mole.name}")

            params_gaussian = self.params_gaussian.copy()
            params_slurm = self.params_slurm.copy()

            params_gaussian.update(
                {
                    "chk": f"{mole.name}.chk",
                    "xyz": str(mole),
                    "titlecard": mole.name,
                    "extra": "",
                }
            )

            params_slurm.update(
                {
                    "jobname": mole.name,
                    "commands": f"g16 {mole.name}.gjf",
                }
            )

            with open(os.path.join(new_folder, f"{mole.name}.gjf"), "w") as f:
                f.write(input_template.substitute(params_gaussian))
            with open(os.path.join(new_folder, f"{mole.name}.sh"), "w") as f:
                f.write(slurm_template.substitute(params_slurm))

        print("Done!")

    def input_gen_psi4(self):
        print("Generating input files for Psi4...")

        input_file = pkg.resource_string("qcds", "templates/psi4.template")
        input_template = Template(input_file.decode("utf-8"))
        slurm_file = pkg.resource_string("qcds", "templates/slurm.template")
        slurm_template = Template(slurm_file.decode("utf-8"))

        for mole in self.mole_configs:
            new_folder = self.input_path
            # self.mole_paths.update({mole.name: new_folder})
            print(f"> {mole.name}")

            params_psi4 = self.params_psi4.copy()
            params_slurm = self.params_slurm.copy()

            params_psi4.update(
                {
                    "xyz": str(mole),
                    "reference": "uks" if (mole.get_spin() > 1) else "rks",
                }
            )

            params_slurm.update(
                {
                    "jobname": mole.name,
                    "commands": f"python {mole.name}.py > {mole.name}.log",
                }
            )

            with open(os.path.join(new_folder, f"{mole.name}.py"), "w") as f:
                f.write(input_template.substitute(params_psi4))
            with open(os.path.join(new_folder, f"{mole.name}.sh"), "w") as f:
                f.write(slurm_template.substitute(params_slurm))

        print("Done")

    def input_gen(self, software: str, path: str, prefix: str, suffix: str):
        self.mole_configs = self.read_moleconfigs(self.geom_path, self.list_file)

        self.input_path = os.path.join(
            path, f"{prefix}_{software}_{self.name}_{suffix}"
        )
        if not os.path.exists(self.input_path):
            os.makedirs(self.input_path)

        match software.lower():
            case "abacus":
                self.input_gen_abacus()
            case "pyscf":
                self.input_gen_pyscf()
            case "gaussian":
                self.input_gen_gaussian()
            case "psi4":
                self.input_gen_psi4()
            case _:
                raise ValueError("Software not supported yet")

    def output_read_abacus(self):
        for mole in self.mole_configs:
            out_folder = os.path.join(self.out_path, mole.name)
            with open(
                os.path.join(
                    out_folder,
                    f"OUT.abacus/running_{self.params_abacus_input["calculation"]}.log",
                ),
                "r",
            ) as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith(" !FINAL_ETOT_IS "):
                    eng = float(line.split()[1])
                    self.mole_eng.update({mole.name: EngUnit(eng, unit="eV")})

    def output_read_pyscf(self):
        for mole in self.mole_configs:
            out_folder = self.out_path
            with open(os.path.join(out_folder, f"{mole.name}.out"), "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("converged SCF energy = "):
                    eng = float(line.split()[4])
                    self.mole_eng.update({mole.name: EngUnit(eng, unit="Hartree")})

    def output_read_gaussian(self):
        for mole in self.mole_configs:
            out_folder = self.out_path
            with open(os.path.join(out_folder, f"{mole.name}.out"), "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith(" SCF Done:"):
                    eng = float(line.split()[4])
                    self.mole_eng.update({mole.name: EngUnit(eng, unit="Hartree")})

    def output_read_psi4(self):
        for mole in self.mole_configs:
            out_folder = self.out_path
            with open(os.path.join(out_folder, f"{mole.name}.out"), "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("    Total Energy = "):
                    eng = float(line.split()[3])
                    self.mole_eng.update({mole.name: EngUnit(eng, unit="Hartree")})

    def output_read(self, software: str, path: str, prefix: str, suffix: str):
        self.out_path = os.path.join(path, f"{prefix}_{software}_{self.name}_{suffix}")
        if not os.path.exists(self.out_path):
            raise FileNotFoundError(f"Output folder {self.out_path} not found")

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