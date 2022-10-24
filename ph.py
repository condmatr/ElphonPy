import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elphonpy.pseudo import get_pseudos
from elphonpy.pw import get_ibrav_celldm, PWInput
from elphonpy.bands import get_simple_kpath

def phonon_input_gen(prefix, structure, pseudo_dict, param_dict_scf, param_dict_ph, multE=1.0, workdir='./phonons', copy_pseudo=False):
    """
    Prepares input file for QE Phonons calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for SCF + PH calculations.
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use. Default to None.
        param_dict_scf (dict): A dict containing sections for SCF calculation input file ('system','control','electrons','kpoint_grid').
        param_dict_ph  (dict): A dict containing sections for SCF calculation input file ('inputph').
        multE (float): Multiplier for pseudopotentials ecutwfc, ecutrho if not specified in scf_param_dict.
        workdir (str): target directory for writing SCF + PH input files. (default: './phonons').
        copy_pseudo (bool): Whether to copy pseudopotentials to current working directory in folder "pseudo". (default: False).
    """
    try:
        os.mkdir(f'{workdir}')
    except OSError as error:
        print(error)
        
    pmd_scf = param_dict_scf
    if 'celldm(1)' and 'ibrav' not in pmd_scf['system'].keys():
        celldm_dict = get_ibrav_celldm(structure)
        pmd_scf['system'].update(celldm_dict)
        
    pseudopotentials, min_ecutwfc, min_ecutrho = get_pseudos(structure, pseudo_dict, copy_pseudo=copy_pseudo)
    
    pmd_scf['control'].update({'prefix':str(prefix).lower()})
    
    if 'ecutwfc' not in pmd_scf['system'].keys():
        pmd_scf['system'].update({'ecutwfc':min_ecutwfc*multE})
    if 'ecutrho' not in pmd_scf['system'].keys():
        pmd_scf['system'].update({'ecutrho':min_ecutrho*multE})
    
    scf_calc = PWInput(structure=structure, pseudo=pseudopotentials,
                               control=pmd_scf['control'], electrons=pmd_scf['electrons'],
                               system=pmd_scf['system'],kpoints_grid=pmd_scf['kpoint_grid'])
    
    scf_calc.write_file(f'{workdir}/{prefix}_scf.in')
    print('SCF input file written to ./phonons')
    
    pmd_ph = param_dict_ph
    
    pmd_ph['inputph'].update({'prefix':f'{str(prefix).lower()}',
                               'fildyn':f'{str(prefix).lower()}.dyn'})
    
    with open(f'./phonons/{prefix}_ph.in', 'w+') as f:
        f.write('&inputph\n')
        for item in param_dict_ph['inputph'].items():
            f.write(f'  {item[0]}={to_str(item[1])}' + ',\n')
        f.write(' /\n')
    
    print(f'PH input file written to {workdir}')
    
def q2r(prefix, qlist, workdir='./phonons'):
    """
    Prepares input file for QE q2r.x calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for SCF + PH calculations.
        qlist (list): list of q points used in phonon calculations (i.e. [nq1,nq2,nq3] from ph.in).
        workdir (str): path to phonon calculation directory from current working directory.
        
    Returns: 
        fc_file_str (str): Name of force constants output file after running q2r.x.
    """
    fc_postfix = ''

    for i in range(len(qlist)):
        fc_postfix = fc_postfix + str(qlist[i])
    
    q2r_dict = {'fildyn':f"'{str.lower(prefix)}.dyn'",
               'zasr':"'simple'",
               'flfrc':f"'{str.lower(prefix)}.{fc_postfix}.fc'"
               }
    
    fc_file_str = q2r_dict['flfrc']
    
    print(f'Saving {workdir}/q2r.in file for calculation of force constants, this will output force constants file {workdir}/{fc_file_str}')
    
    out = []

    for i in list(q2r_dict.keys()):
        out.append(f" {i}={q2r_dict[i]}")
        
    filename = f'{workdir}/q2r.in'
    with open(filename, "w+") as f:
        f.write('&input' + '\n')
        for i in out:
            f.write(i + ',\n')
        f.write('/' + '\n')
    return fc_file_str
    
def matdyn(prefix, structure, pseudo_dict, fc_file_str, workdir='phonons'):
    
    """
    Prepares input file for QE matdyn.x calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for SCF + PH calculations.
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use.
        fc_file_str (str): Name of force constants output file after running q2r.x.
        workdir (str): path to phonon calculation directory from current working directory.
        
    Returns: 
        amass_dict (dict): Dictionary of atomic masses.
        qp_dict (dict): q-path dictionary.
    """
    
    from pymatgen.core.composition import Element
    from pymatgen.symmetry.kpath import KPathSeek as seekpath
    
    pseudopotentials = get_pseudos(structure, pseudo_dict, copy_pseudo=False)
    pseudopotentials = pseudopotentials[0]
    atoms = sorted(list(pseudopotentials.keys()))
    
    amass_dict = dict()
    
    for i in range(len(atoms)):
        imass = Element(atoms[i]).atomic_mass
        imass_dict = {f'amass({i+1})':f'{str(imass)[:-3]}'}
        amass_dict.update(imass_dict)

    matdyn_dict = {'asr':'simple',
                   'flfrq':f'{str.lower(prefix)}.freq'
                  }
    
    phonon_file_str = matdyn_dict['flfrq']
    
    qp_dict = get_simple_kpath(structure)
    
    qpoints_out = qp_dict['kpoints']
    
    
    out = []
    
    out.append(f' flfrc={fc_file_str}')
    
    for i in list(matdyn_dict.keys()):
        out.append(f' {i}={to_str(matdyn_dict[i])}')
    
    for i in list(amass_dict.keys()):
        out.append(f' {i}={amass_dict[i]}')
    
    filename = f'{workdir}/matdyn.in'
    with open(filename, "w") as f:
        f.write('&input' + '\n')
        for i in out:
            f.write(i + ',\n')
        f.write('/' + '\n')
        f.write(str(len(kp_out)) + '\n')
        for i in range(len(kp_out)):
            qpi = qpoints_out[i]
            f.write(f'    {qpi[0]:.10f}  {qpi[1]:.10f}  {qpi[2]:.10f}' + ' 1\n')
    
    print(f'Saving matdyn.in file for calculation of phonon dispersion, this will output the phonon dispersion file {phonon_file_str}')
    return amass_dict, qp_dict 
            
def plot_phonons(prefix, qp_dict):
    """
    Prepares input file for QE matdyn.x calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for SCF + PH calculations.
        qp_dict (dict): q-path dictionary.
        
    Returns: 
        phonons_dataframe (pandas.DataFrame): The data which the phonons are plotted.
    """
    df = pd.read_csv(f'phonons/{str.lower(prefix)}.freq.gp', delim_whitespace=True, header=None)
    col_names = ['High_sym']

    high_sym_symbol = qp_dict['path_symbols']
    high_sym_idx = qp_dict['path_idx_wrt_kpt']

    rng = np.arange(1, int(len(list(df))))
    for i in rng:
        col_names.append(f'Mode_{i}')
    df.columns = col_names

    fig, ax = plt.subplots(figsize=[4,3], dpi=300)
    
    mini = np.min(df.values)
    maxi = np.max(df.values)
    if mini > -10:
        miny = -10
    else:
        miny = mini+(0.2*mini)
    maxy = maxi + (0.2*maxi)
    
    ax.axhline(0, xmin=0, xmax=max(df['High_sym']), c='k', ls='--', lw=0.5, alpha=0.5)
    j=0
    for i in range(len(df['High_sym'])):
        if i in high_sym_idx:
            ax.vlines(df['High_sym'].iloc[i], ymin=miny, ymax=maxy, lw=0.3, colors='k')
            ax.text(df['High_sym'].iloc[i], -0.05, f'{high_sym_symbol[j]}', ha='center', va='center', transform=ax.transAxes)
            j+=1

    for i in rng:
        ax.plot(df['High_sym'], df[f'Mode_{i}'], c='r', lw=0.5)
    
    ax.ylim(miny,maxy)
    ax.xlim(0,max(df['High_sym']))
    ax.xticks([])

    fig.savefig(f'phonons/{prefix}_phonons.png')
    
    return df
