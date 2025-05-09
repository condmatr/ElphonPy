import os
import subprocess
import re
import shutil
import json
import numpy as np
import math
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import IStructure
from elphonpy.pseudo import get_pseudos
from collections import defaultdict
from monty.io import zopen
from monty.re import regrep
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.util.io_utils import clean_lines

# A portion of these functions were adapted from pymatgen.io.pwscf, whose author is Shyue Ping Ong. 

def angs_to_bohr(lattice_param):
    return lattice_param*1.88973

def get_ibrav_celldm(structure, get_primitive=True):
    """
    Return the QE ibrav parameter for the space group of the system and define necessary lattice parameters in bohr radii to interface with QE
    (NOTE: User may need to define ibrav themselves if crystal system is not included in these options/weird structure)

    Args:
        structure (Pymatgen Structure or IStructure): Input structure.
        get_primitive (bool): Whether to represent structure as primitive into function.
        
    Returns:
        ibrav_celldm_dictionary (dict): Dictionary containing ibrav and necessary celldm parameters for QE input files.
    """

    sga = SpacegroupAnalyzer(structure)
    
    if get_primitive == True:
        structure = sga.find_primitive()
        
    crys_sys = sga.get_crystal_system()
    sg_sym = sga.get_space_group_symbol()
    lat = structure.lattice
    dict_ = dict()
    print("Trying ibrav identification: System = ", crys_sys, sg_sym[0])

    if crys_sys == 'cubic' and sg_sym[0] == 'P':
        dict_ = {'ibrav':1,
                 'celldm(1)':angs_to_bohr(lat.a)
                }

    elif crys_sys == 'cubic' and sg_sym[0] == 'F':
        a = 2 * np.abs(lat.matrix[0][0])
        dict_ = {'ibrav':2,
                 'celldm(1)':angs_to_bohr(a)
                }

    elif crys_sys == 'cubic' and sg_sym[0] == 'I':
        a = 2 * np.abs(lat.matrix[0][0])
        dict_ = {'ibrav':3,
                 'celldm(1)':angs_to_bohr(a)
                }

    elif crys_sys == 'hexagonal':
        dict_ = {'ibrav':4,
                 'celldm(1)':angs_to_bohr(lat.a),
                 'celldm(3)':lat.c/lat.a
                }

    elif crys_sys == 'tetragonal' and sg_sym[0] == 'P':
        dict_ = {'ibrav':6,
                 'celldm(1)':angs_to_bohr(lat.a),
                 'celldm(3)':lat.c/lat.a
                }
    
    elif crys_sys == 'tetragonal' and sg_sym[0] == 'I':
        dict_ = {'ibrav':7,
                 'celldm(1)':angs_to_bohr(lat.a),
                 'celldm(3)':lat.c/lat.a
                }
    
    elif crys_sys == 'trigonal' and sg_sym[0] == 'P':
        dict_ = {'ibrav':4,
                 'celldm(1)':angs_to_bohr(lat.a),
                 'celldm(3)':lat.c/lat.a
                }

    elif crys_sys == 'triclinic':
        angles = lat.angles
        dict_ = {'ibrav':14,
                 'celldm(1)':angs_to_bohr(lat.a),
                 'celldm(2)': lat.b/lat.a,
                 'celldm(3)': lat.c/lat.a,
                 'celldm(4)': np.cos(angles[0]*(np.pi/180)),
                 'celldm(5)': np.cos(angles[0]*(np.pi/180)),
                 'celldm(6)': np.cos(angles[0]*(np.pi/180))}

    else:
        print("Your system's ibrav is not built into the software yet, ibrav is set to 0.")
        dict_ = {'ibrav':0}
            
    return dict_


def get_cell_params(structure):
    
    """
    Returns cell_parameters of input structure in angstrom, eg:
    
    CELL_PARAMETERS angstrom
       3.337812083   0.000000000   0.000000000
       0.000000000   6.714860814   0.033113024
       0.000000000  -0.004333785  16.877820267
       
    Args:
    structure (Pymatgen Structure or IStructure): Input structure.
    
    Returns:
    out (list): list of strings for each line required in the CELL_PARAMETERS card for QE input.
    """
    
    out = ["CELL_PARAMETERS angstrom"]
    mat = structure.lattice.matrix
    for i in range(3):
        out.append(f"{mat[i,0]:>18.10f}{mat[i,1]:>18.10f}{mat[i,2]:>18.10f}")
        
    return out

def to_str(v):
    """
    Conversion of scientific notation and booleans to QE scientific notation and booleans.
    (i.e. e --> d, True --> .TRUE.)
    """
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, float):
        return f"{str(v).replace('e', 'd')}"
    if isinstance(v, bool):
        if v:
            return ".TRUE."
        return ".FALSE."
    return v


class PWInput:
    """
    Base input file class.
    """

    def __init__(
        self,
        structure,
        pseudo=None,
        control=None,
        system=None,
        electrons=None,
        ions=None,
        cell=None,
        kpoints_mode="automatic",
        kpoints_grid=(1, 1, 1),
        kpoints_shift=(0, 0, 0),
    ):
        """
        Initializes a PWSCF input file.
        Args:
            structure (Structure): Input structure. For spin-polarized calculation,
                properties (e.g. {"starting_magnetization": -0.5,
                "pseudo": "Mn.pbe-sp-van.UPF"}) on each site is needed instead of
                pseudo (dict).
            pseudo (dict): A dict of the pseudopotentials to use. Default to None.
            control (dict): Control parameters. Refer to official PWSCF doc
                on supported parameters. Default to {"calculation": "scf"}
            system (dict): System parameters. Refer to official PWSCF doc
                on supported parameters. Default to None, which means {}.
            electrons (dict): Electron parameters. Refer to official PWSCF doc
                on supported parameters. Default to None, which means {}.
            ions (dict): Ions parameters. Refer to official PWSCF doc
                on supported parameters. Default to None, which means {}.
            cell (dict): Cell parameters. Refer to official PWSCF doc
                on supported parameters. Default to None, which means {}.
            kpoints_mode (str): Kpoints generation mode. Default to automatic.
            kpoints_grid (sequence): The kpoint grid. Default to (1, 1, 1).
            kpoints_shift (sequence): The shift for the kpoints. Defaults to
                (0, 0, 0).
        """
        self.structure = structure
        sections = {}
        sections["control"] = control or {"calculation": "scf"}
        sections["system"] = system or {}
        sections["electrons"] = electrons or {}
        sections["ions"] = ions or {}
        sections["cell"] = cell or {}
        if pseudo is None:
            for site in structure:
                try:
                    site.properties["pseudo"]
                except KeyError:
                    raise PWInputError(f"Missing {site} in pseudo specification!")
        else:
            for species in self.structure.composition.keys():
                if str(species) not in pseudo:
                    raise PWInputError(f"Missing {species} in pseudo specification!")
        self.pseudo = pseudo

        self.sections = sections
        self.kpoints_mode = kpoints_mode
        self.kpoints_grid = kpoints_grid
        self.kpoints_shift = kpoints_shift

    def __str__(self):
        out = []
        site_descriptions = {}

        if self.pseudo is not None:
            site_descriptions = self.pseudo
        else:
            c = 1
            for site in self.structure:
                name = None
                for k, v in site_descriptions.items():
                    if site.properties == v:
                        name = k

                if name is None:
                    name = site.specie.symbol + str(c)
                    site_descriptions[name] = site.properties
                    c += 1

        def to_str(v):
            if isinstance(v, str):
                return f"'{v}'"
            if isinstance(v, float):
                return f"{str(v).replace('e', 'd')}"
            if isinstance(v, bool):
                if v:
                    return ".TRUE."
                return ".FALSE."
            return v

        for k1 in ["control", "system", "electrons", "ions", "cell"]:
            v1 = self.sections[k1]
            out.append(f"&{k1.upper()}")
            sub = []
            for k2 in sorted(v1.keys()):
                if isinstance(v1[k2], list):
                    n = 1
                    for l in v1[k2][: len(site_descriptions)]:
                        sub.append(f"  {k2}({n}) = {to_str(v1[k2][n - 1])}")
                        n += 1
                else:
                    sub.append(f"  {k2} = {to_str(v1[k2])}")
            if k1 == "system":
                if "ibrav" not in self.sections[k1]:
                    sub.append("  ibrav = 0")
                if "nat" not in self.sections[k1]:
                    sub.append(f"  nat = {len(self.structure)}")
                if "ntyp" not in self.sections[k1]:
                    sub.append(f"  ntyp = {len(site_descriptions)}")
            sub.append("/")
            out.append(",\n".join(sub))

        out.append("ATOMIC_SPECIES")
        for k, v in sorted(site_descriptions.items(), key=lambda i: i[0]):
            e = re.match(r"[A-Z][a-z]?", k).group(0)
            if self.pseudo is not None:
                p = v
            else:
                p = v["pseudo"]
            out.append(f"  {k}  {Element(e).atomic_mass:.4f} {p}")

        out.append("ATOMIC_POSITIONS crystal")
        if self.pseudo is not None:
            for site in self.structure:
                out.append(f"  {site.specie} {site.a:.12f} {site.b:.12f} {site.c:.12f}")
        else:
            for site in self.structure:
                name = None
                for k, v in sorted(site_descriptions.items(), key=lambda i: i[0]):
                    if v == site.properties:
                        name = k
                out.append(f"  {name} {site.a:.12f} {site.b:.12f} {site.c:.12f}")

        out.append(f"K_POINTS {self.kpoints_mode}")
        if self.kpoints_mode == "automatic":
            kpt_str = [f"{i}" for i in self.kpoints_grid]
            kpt_str.extend([f"{i}" for i in self.kpoints_shift])
            out.append(f"  {' '.join(kpt_str)}")
        elif self.kpoints_mode == "crystal_b":
            out.append(f" {str(len(self.kpoints_grid))}")
            for i in range(len(self.kpoints_grid)):
                kpt_str = [f"{entry:.4f}" for entry in self.kpoints_grid[i]]
                out.append(f" {' '.join(kpt_str)}")
        elif self.kpoints_mode == "gamma":
            pass
    
        if "ibrav" in self.sections["system"].keys():
            if self.sections["system"]["ibrav"] != 0:
                pass
            if self.sections["system"]["ibrav"] == 0:
                for line in get_cell_params(self.structure):
                    out.append(line)
            return "\n".join(out)
    def as_dict(self):
        """
        Create a dictionary representation of a PWInput object
        Returns:
            dict
        """
        pwinput_dict = {
            "structure": self.structure.as_dict(),
            "pseudo": self.pseudo,
            "sections": self.sections,
            "kpoints_mode": self.kpoints_mode,
            "kpoints_grid": self.kpoints_grid,
            "kpoints_shift": self.kpoints_shift,
        }
        return pwinput_dict

    @classmethod
    def from_dict(cls, pwinput_dict):
        """
        Load a PWInput object from a dictionary.
        Args:
            pwinput_dict (dict): dictionary with PWInput data
        Returns:
            PWInput object
        """
        pwinput = cls(
            structure=Structure.from_dict(pwinput_dict["structure"]),
            pseudo=pwinput_dict["pseudo"],
            control=pwinput_dict["sections"]["control"],
            system=pwinput_dict["sections"]["system"],
            electrons=pwinput_dict["sections"]["electrons"],
            ions=pwinput_dict["sections"]["ions"],
            cell=pwinput_dict["sections"]["cell"],
            kpoints_mode=pwinput_dict["kpoints_mode"],
            kpoints_grid=pwinput_dict["kpoints_grid"],
            kpoints_shift=pwinput_dict["kpoints_shift"],
        )
        return pwinput

    def write_file(self, filename):
        """
        Write the PWSCF input file.
        Args:
            filename (str): The string filename to output to.
        """
        with open(filename, "w") as f:
            f.write(self.__str__())

    @staticmethod
    def from_file(filename):
        """
        Reads an PWInput object from a file.
        Args:
            filename (str): Filename for file
        Returns:
            PWInput object
        """
        with zopen(filename, "rt") as f:
            return PWInput.from_string(f.read())

    @staticmethod
    def from_string(string):
        """
        Reads an PWInput object from a string.
        Args:
            string (str): PWInput string
        Returns:
            PWInput object
        """
        lines = list(clean_lines(string.splitlines()))

        def input_mode(line):
            if line[0] == "&":
                return ("sections", line[1:].lower())
            if "ATOMIC_SPECIES" in line:
                return ("pseudo",)
            if "K_POINTS" in line:
                return "kpoints", line.split()[1]
            if "OCCUPATIONS" in line:
                return "occupations"
            if "CELL_PARAMETERS" in line or "ATOMIC_POSITIONS" in line:
                return "structure", line.split()[1]
            if line == "/":
                return None
            return mode

        sections = {
            "control": {},
            "system": {},
            "electrons": {},
            "ions": {},
            "cell": {},
        }
        pseudo = {}
        lattice = []
        species = []
        coords = []
        structure = None
        site_properties = {"pseudo": []}
        mode = None
        for line in lines:
            mode = input_mode(line)
            if mode is None:
                pass
            elif mode[0] == "sections":
                section = mode[1]
                m = re.match(r"(\w+)\(?(\d*?)\)?\s*=\s*(.*)", line)
                if m:
                    key = m.group(1).strip()
                    key_ = m.group(2).strip()
                    val = m.group(3).strip()
                    if key_ != "":
                        if sections[section].get(key, None) is None:
                            val_ = [0.0] * 20  # MAX NTYP DEFINITION
                            val_[int(key_) - 1] = PWInput.proc_val(key, val)
                            sections[section][key] = val_

                            site_properties[key] = []
                        else:
                            sections[section][key][int(key_) - 1] = PWInput.proc_val(key, val)
                    else:
                        sections[section][key] = PWInput.proc_val(key, val)

            elif mode[0] == "pseudo":
                m = re.match(r"(\w+)\s+(\d*.\d*)\s+(.*)", line)
                if m:
                    pseudo[m.group(1).strip()] = m.group(3).strip()
            elif mode[0] == "kpoints":
                m = re.match(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
                if m:
                    kpoints_grid = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    kpoints_shift = (int(m.group(4)), int(m.group(5)), int(m.group(6)))
                else:
                    kpoints_mode = mode[1]
                    kpoints_grid = (1, 1, 1)
                    kpoints_shift = (0, 0, 0)

            elif mode[0] == "structure":
                m_l = re.match(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", line)
                m_p = re.match(r"(\w+)\s+(-?\d+\.\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", line)
                if m_l:
                    lattice += [
                        float(m_l.group(1)),
                        float(m_l.group(2)),
                        float(m_l.group(3)),
                    ]
                elif m_p:
                    site_properties["pseudo"].append(pseudo[m_p.group(1)])
                    species.append(m_p.group(1))
                    coords += [[float(m_p.group(2)), float(m_p.group(3)), float(m_p.group(4))]]

                    if mode[1] == "angstrom":
                        coords_are_cartesian = True
                    elif mode[1] == "crystal":
                        coords_are_cartesian = False
        structure = Structure(
            Lattice(lattice),
            species,
            coords,
            coords_are_cartesian=coords_are_cartesian,
            site_properties=site_properties,
        )
        return PWInput(
            structure=structure,
            control=sections["control"],
            pseudo=pseudo,
            system=sections["system"],
            electrons=sections["electrons"],
            ions=sections["ions"],
            cell=sections["cell"],
            kpoints_mode=kpoints_mode,
            kpoints_grid=kpoints_grid,
            kpoints_shift=kpoints_shift,
        )

    @staticmethod
    def proc_val(key, val):
        """
        Static helper method to convert PWINPUT parameters to proper type, e.g.,
        integers, floats, etc.
        Args:
            key: PWINPUT parameter key
            val: Actual value of PWINPUT parameter.
        """
        float_keys = (
            "etot_conv_thr",
            "forc_conv_thr",
            "conv_thr",
            "Hubbard_U",
            "Hubbard_J0",
            "defauss",
            "starting_magnetization",
        )

        int_keys = (
            "nstep",
            "iprint",
            "nberrycyc",
            "gdir",
            "nppstr",
            "ibrav",
            "nat",
            "ntyp",
            "nbnd",
            "nr1",
            "nr2",
            "nr3",
            "nr1s",
            "nr2s",
            "nr3s",
            "nspin",
            "nqx1",
            "nqx2",
            "nqx3",
            "lda_plus_u_kind",
            "edir",
            "report",
            "esm_nfit",
            "space_group",
            "origin_choice",
            "electron_maxstep",
            "mixing_ndim",
            "mixing_fixed_ns",
            "ortho_para",
            "diago_cg_maxiter",
            "diago_david_ndim",
            "nraise",
            "bfgs_ndim",
            "if_pos",
            "nks",
            "nk1",
            "nk2",
            "nk3",
            "sk1",
            "sk2",
            "sk3",
            "nconstr",
        )

        bool_keys = (
            "wf_collect",
            "tstress",
            "tprnfor",
            "lkpoint_dir",
            "tefield",
            "dipfield",
            "lelfield",
            "lorbm",
            "lberry",
            "lfcpopt",
            "monopole",
            "nosym",
            "nosym_evc",
            "noinv",
            "no_t_rev",
            "force_symmorphic",
            "use_all_frac",
            "one_atom_occupations",
            "starting_spin_angle",
            "noncolin",
            "x_gamma_extrapolation",
            "lda_plus_u",
            "lspinorb",
            "london",
            "ts_vdw_isolated",
            "xdm",
            "uniqueb",
            "rhombohedral",
            "realxz",
            "block",
            "scf_must_converge",
            "adaptive_thr",
            "diago_full_acc",
            "tqr",
            "remove_rigid_rot",
            "refold_pos",
        )

        def smart_int_or_float(numstr):
            if numstr.find(".") != -1 or numstr.lower().find("e") != -1:
                return float(numstr)
            return int(numstr)

        try:
            if key in bool_keys:
                if val.lower() == ".true.":
                    return True
                if val.lower() == ".false.":
                    return False
                raise ValueError(key + " should be a boolean type!")

            if key in float_keys:
                return float(re.search(r"^-?\d*\.?\d*d?-?\d*", val.lower()).group(0).replace("d", "e"))

            if key in int_keys:
                return int(re.match(r"^-?[0-9]+", val).group(0))

        except ValueError:
            pass

        try:
            val = val.replace("d", "e")
            return smart_int_or_float(val)
        except ValueError:
            pass

        if "true" in val.lower():
            return True
        if "false" in val.lower():
            return False

        m = re.match(r"^[\"|'](.+)[\"|']$", val)
        if m:
            return m.group(1)
        
def kpt_res_grid(structure, res):
    """
    Calculates kpoint grid resolution in units 1/A.
    
    Args:
        structure (Pymatgen Structure or IStructure): Input structure.
        res (float) : Desired kpoint resolution (1/A).
    
    Returns:
        kpoint_grid (list) : kpoint grid for given structure and resolution
    """
    if res <= 0:
        raise ValueError("Resolution must be greater than 0.")
    
    i, j, k = structure.lattice.reciprocal_lattice.abc
    initial_grid = [math.ceil(i/res), math.ceil(j/res), math.ceil(k/res)]
    

    sga = SpacegroupAnalyzer(structure)
    crystal_system = sga.get_crystal_system()
    #round_odd = ['hexagonal', 'cubic', 'tetragonal']
    
    #if crystal_system in round_odd:
    #    kpoint_grid = [g+1 if g%2 == 0 else g for g in initial_grid]
    #else:
    kpoint_grid = initial_grid
    
    return kpoint_grid
        
def automatic_kppa(structure, kppa):
    
    """
    Calculates kpoint grid to match kppa resolution. 

    Args:
        structure (Pymatgen Structure or IStructure): Input structure.
        kppa (float): Density of desired kpoint grid.
        
    Returns:
        kpoint_grid (tuple): kpoint grid for given structure and density.
    """
    
    if np.fabs((np.floor(kppa ** (1 / 3) + 0.5)) ** 3 - kppa) < 1:
                kppa += kppa * 0.01
    latt = structure.lattice
    lengths = latt.abc
    ngrid = kppa / structure.num_sites
    mult = (ngrid * lengths[0] * lengths[1] * lengths[2]) ** (1 / 3)

    num_div = [int(np.floor(max(mult / l, 1))) for l in lengths]
    
    return tuple(num_div)
        

def scf_input_gen(prefix, structure, pseudo_dict, param_dict, multE=1,  rhoe=None, workdir='./scf',copy_pseudo=False):
    
    """
    Prepares input file for QE SCF calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for scf calculations.
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use. Default to None.
        param_dict  (dict): A dict containing sections for input file ('system','control','electrons','kpoint_grid')
        multE (float): Multiplier for pseudopotentials ecutwfc, ecutrho if not specified in param_dict.
        workdir (str): target directory for writing SCF input file.
        copy_pseudo (bool): Whether to copy pseudopotentials to current working directory in folder "pseudo".
    """
    
    if workdir != './':
        try:
            os.mkdir(workdir)
        except OSError as error:
            print(error)
        
    pmd = param_dict
    
    pseudopotentials, min_ecutwfc, min_ecutrho = get_pseudos(structure, pseudo_dict, copy_pseudo=copy_pseudo)
        
    if 'ibrav' not in pmd['system'].keys() and 'celldm(1)' not in pmd['system'].keys():
        celldm_dict = get_ibrav_celldm(structure)
        pmd['system'].update(celldm_dict)
    
    if pmd['system']['ibrav'] == 0:
        cell_params = get_cell_params(structure)
    
    if 'ecutwfc' not in pmd['system'].keys():
        pmd['system'].update({'ecutwfc':min_ecutwfc*multE})
    if 'ecutrho' not in pmd['system'].keys():
        pmd['system'].update({'ecutrho':min_ecutrho*multE})
        if rhoe != None:
            pmd['system'].update({'ecutrho':min_ecutwfc*multE*rhoe})

    
    scf_calc = PWInput(structure=structure, pseudo=pseudopotentials, control=pmd['control'],
                       electrons=pmd['electrons'], system=pmd['system'], cell=None,
                       kpoints_grid=pmd['kpoint_grid'], ions=None)
    
    scf_calc.write_file(f'./{workdir}/{prefix}_scf.in')
    
    print(f'SCF input file written to {workdir}')

def nscf_input_gen(prefix, structure, pseudo_dict, param_dict, multE=1, rhoe=None, workdir='./nscf', copy_pseudo=False):
    """
    Prepares input file for QE NSCF calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for scf calculations.
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use. Default to None.
        param_dict  (dict): A dict containing sections for input file ('system','control','electrons','kpoint_grid')
        multE (float): Multiplier for pseudopotentials ecutwfc, ecutrho if not specified in param_dict.
        workdir (str): target directory for writing SCF input file.
        copy_pseudo (bool): Whether to copy pseudopotentials to current working directory in folder "pseudo".
    """
    if workdir != './':
        try:
            os.mkdir(workdir)
        except OSError as error:
            print(error)
        
    pmd = param_dict
    
    pseudopotentials, min_ecutwfc, min_ecutrho = get_pseudos(structure, pseudo_dict, copy_pseudo=copy_pseudo)
    
    if 'ibrav' not in pmd['system'].keys() and 'celldm(1)' not in pmd['system'].keys():
        celldm_dict = get_ibrav_celldm(structure)
        pmd['system'].update(celldm_dict)
    
    if pmd['system']['ibrav'] == 0:
        cell_params = get_cell_params(structure)
    
    if 'ecutwfc' not in pmd['system'].keys():
        pmd['system'].update({'ecutwfc':min_ecutwfc*multE})
    if 'ecutrho' not in pmd['system'].keys():
        pmd['system'].update({'ecutrho':min_ecutrho*multE})
        if rhoe != None:
            pmd['system'].update({'ecutrho':min_ecutwfc*multE*rhoe})
    
    if 'kpoint_mode' in pmd.keys() and pmd['kpoint_mode'] == 'automatic':
        print('automatic k-points selected')
        nscf_calc = PWInput(structure=structure, pseudo=pseudopotentials,
                            control=pmd['control'], electrons=pmd['electrons'],
                            system=pmd['system'],kpoints_mode=pmd['kpoint_mode'],
                            kpoints_grid=pmd['kpoint_grid'])
        nscf_calc.write_file(f'{workdir}/{prefix}_nscf.in')
    
    else:
        nscf_calc =  PWInput(structure=structure, pseudo=pseudopotentials,
                             control=pmd['control'], electrons=pmd['electrons'],
                             system=pmd['system'])
        nscf_calc.write_file(f'{workdir}/nscf.temp')

        with open(f'{workdir}/nscf.temp', 'r+') as f:
            temp = f.readlines()
            for i, line in enumerate(temp):
                if "K_POINTS" in line:
                    temp = temp[:i] + temp[i+2:]
        f.close()

        def dense_k(kg0, kg1, kg2):

            tot = kg0*kg1*kg2
            dense_k = []
            for ii in range(kg0):
                for jj in range(kg1):
                    for kk in range(kg2):
                        dense_k.append([ii/kg0,jj/kg1,kk/kg2])

            return dense_k, tot

        dense_k_grid, total_k = dense_k(pmd['kpoint_grid'][0], pmd['kpoint_grid'][1], pmd['kpoint_grid'][2])

        with open(f'{workdir}/{prefix}_nscf.in', 'w') as f:
            for i in temp:
                f.write(i)

            f.write('\nK_POINTS crystal\n')
            f.write(f'{total_k}\n')
            for i in dense_k_grid:
                f.write(f'  {i[0]:.8f}  {i[1]:.8f}  {i[2]:.8f}  {1/total_k:.4e}\n')
        f.close()
    
    print(f'NSCF input file written to {workdir}')
    
def relax_input_gen(prefix, structure, pseudo_dict, param_dict, multE=1,  rhoe=None, workdir='./relax', copy_pseudo=False):
    """
    Prepares input file for QE SCF Relax calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for scf calculations.
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use. Default to None.
        param_dict  (dict): A dict containing sections for input file ('system','control','electrons','kpoint_grid')
        multE (float): Multiplier for pseudopotentials ecutwfc, ecutrho if not specified in param_dict.
        workdir (str): target directory for writing SCF input file.
        copy_pseudo (bool): Whether to copy pseudopotentials to current working directory in folder "pseudo".
    """
    if workdir != './':
        try:
            os.makedirs(workdir)
        except OSError as error:
            print(error)
        
    pmd = param_dict
    
    pseudopotentials, min_ecutwfc, min_ecutrho = get_pseudos(structure, pseudo_dict, copy_pseudo=copy_pseudo)
    if 'ibrav' not in pmd['system'].keys() and 'celldm(1)' not in pmd['system'].keys():
        celldm_dict = get_ibrav_celldm(structure)
        pmd['system'].update(celldm_dict)
        
    if pmd['system']['ibrav'] == 0:
        cell_params = get_cell_params(structure)
    
    if 'ecutwfc' not in pmd['system'].keys():
        pmd['system'].update({'ecutwfc':min_ecutwfc*multE})
    if 'ecutrho' not in pmd['system'].keys():
        pmd['system'].update({'ecutrho':min_ecutrho*multE})
        if rhoe != None:
            pmd['system'].update({'ecutrho':min_ecutwfc*multE*rhoe})
    
    relax_calc = PWInput(structure=structure, pseudo=pseudopotentials, control=pmd['control'],
                         electrons=pmd['electrons'], system=pmd['system'], cell=pmd['cell'],
                         kpoints_grid=pmd['kpoint_grid'], ions=pmd['ions'])
    
    relax_calc.write_file(f'./{workdir}/{prefix}_relax.in')
    
def read_relax_output(prefix, workdir='./relax', out_filename=None, cif_dir=None, get_primitive=True):
    import re
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    """
    Converts a vcrelax.out file to a CIF file.
    
    This function handles two cases:
      - When the file contains "CELL_PARAMETERS (angstrom)" (ibrav=0), the values are
        already in angstrom.
      - When the file contains "CELL_PARAMETERS (alat= ...)" (ibrav != 0), the numbers are
        in relative units. The number following "alat=" (given in bohr radii) is used to convert
        these values to angstrom (using 1 bohr ≈ 0.52917721067 Å).
    
    Args:
        prefix (str): Prefix of input/output files for relax calculations.
        workdir (str): Path to the relax directory (default: './relax').
        out_filename (str): Filename for the output CIF file (default: prefix).
        cif_dir (str): Directory where the CIF file will be saved (default: workdir).
        get_primitive (bool): Whether to represent the structure in CIF as its primitive cell (default: True).
        
    Returns:
        structure (pymatgen.core.structure.Structure): The Pymatgen structure object from the relax calculation.
    """
    
    # Set output filename and directory
    filename = prefix if out_filename is None else out_filename
    save_cif_dir = workdir if cif_dir is None else cif_dir
    
    file_path = f"{workdir}/{prefix}_relax.out"
    
    # Initialize lists for lattice vectors and atomic positions
    cell_parameters = []
    atomic_positions = []
    species = []
    
    # Prepare regular expressions
    # This regex captures the content inside parentheses in the CELL_PARAMETERS header.
    cell_param_regex = re.compile(r'CELL_PARAMETERS\s*\((.*?)\)', re.IGNORECASE)
    atomic_pos_regex = re.compile(r'ATOMIC_POSITIONS\s*\(crystal\)', re.IGNORECASE)
    
    # Read the file contents
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Flags to control parsing of sections
    in_section = False
    cell_params_started = False
    atomic_pos_started = False
    
    # Default conversion: if values are already in angstrom, no conversion is needed.
    conversion_factor = 1.0  
    bohr_to_angstrom = 0.52917721067
    
    for line in lines:
        if 'Begin final coordinates' in line:
            in_section = True
            continue
        if 'End final coordinates' in line:
            break
        if in_section:
            # Look for the cell parameters header.
            cell_param_header = cell_param_regex.search(line)
            if cell_param_header:
                cell_params_started = True
                atomic_pos_started = False
                header_info = cell_param_header.group(1)
                # If the header contains "alat", perform conversion.
                if 'alat' in header_info:
                    alat_match = re.search(r'alat\s*=\s*([\d\.Ee+-]+)', header_info)
                    if alat_match:
                        alat_value = float(alat_match.group(1))
                        conversion_factor = alat_value * bohr_to_angstrom
                else:
                    # If not "alat", assume the cell parameters are given in angstrom.
                    conversion_factor = 1.0
                continue
            
            # Look for the atomic positions header.
            if atomic_pos_regex.search(line):
                atomic_pos_started = True
                cell_params_started = False
                continue
            
            # Parse the cell parameters lines.
            if cell_params_started:
                parts = line.strip().split()
                if len(parts) == 3:
                    # Multiply each element by the conversion factor.
                    row = [float(x) * conversion_factor for x in parts]
                    cell_parameters.append(row)
                continue
            
            # Parse the atomic positions lines.
            if atomic_pos_started:
                parts = line.strip().split()
                if len(parts) == 4:
                    species.append(parts[0])
                    atomic_positions.append([float(x) for x in parts[1:]])
                continue
    
    # Ensure that both sections were found
    if not cell_parameters:
        raise ValueError("CELL_PARAMETERS not found in the specified section.")
    if not atomic_positions:
        raise ValueError("ATOMIC_POSITIONS not found in the specified section.")
    
    # Create the Lattice object (the matrix is now in angstrom).
    lattice = Lattice(cell_parameters)
    
    # Create the Structure object.
    structure = Structure(
        lattice=lattice,
        species=species,
        coords=atomic_positions,
        coords_are_cartesian=False  # atomic positions are provided in crystal (fractional) coordinates
    )
    
    print(structure)
    
    # Convert to the primitive structure if requested.
    if get_primitive:
        structure = SpacegroupAnalyzer(structure).find_primitive()
    else:
        print('Primitive structure not chosen. Please double check your celldm and ibrav if using this structure in your next calculation.')
    
    # Save the structure to a CIF file.
    structure.to(filename=f'{save_cif_dir}/{filename}.cif')
    
    return structure
    

