import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elphonpy.pseudo import get_pseudos
from elphonpy.pw import get_ibrav_celldm, PWInput, to_str
from elphonpy.bands import get_simple_kpath, join_last_to_first_latex

def phonon_input_gen(prefix, structure, pseudo_dict, param_dict_scf, param_dict_ph, multE=1.0, rhoe=None, workdir='./phonons', copy_pseudo=False):
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
        if rhoe != None:
            pmd['system'].update({'ecutrho':min_ecutwfc*multE*rhoe})
    
    scf_calc = PWInput(structure=structure, pseudo=pseudopotentials,
                               control=pmd_scf['control'], electrons=pmd_scf['electrons'],
                               system=pmd_scf['system'],kpoints_grid=pmd_scf['kpoint_grid'])
    
    scf_calc.write_file(f'{workdir}/{prefix}_scf.in')
    print(f'SCF input file written to {workdir}')
    
    pmd_ph = param_dict_ph
    
    pmd_ph['inputph'].update({'prefix':f'{str(prefix).lower()}',
                               'fildyn':f'{str(prefix).lower()}.dyn'})
    
    with open(f'{workdir}/{prefix}_ph.in', 'w+') as f:
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
    
def matdyn(prefix, structure, kpath_dict, pseudo_dict, fc_file_str, workdir='phonons'):
    
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
    
    pseudopotentials = get_pseudos(structure, pseudo_dict, copy_pseudo=False)
    pseudopotentials = pseudopotentials[0]

    matdyn_dict = {'asr':'simple',
                   'flfrq':f'{str.lower(prefix)}.freq'
                  }
    
    phonon_file_str = matdyn_dict['flfrq']
    
    qp_dict = kpath_dict
    
    qpoints_out = qp_dict['kpoints']
    
    
    out = []
    
    out.append(f' flfrc={fc_file_str}')
    out.append(f' q_in_cryst_coord=.TRUE.')
    for i in list(matdyn_dict.keys()):
        out.append(f' {i}={to_str(matdyn_dict[i])}')
    
    filename = f'{workdir}/matdyn.in'
    with open(filename, "w") as f:
        f.write('&input' + '\n')
        for i in out:
            f.write(i + ',\n')
        f.write('/' + '\n')
        f.write(str(len(qpoints_out)) + '\n')
        for i in range(len(qpoints_out)):
            qpi = qpoints_out[i]
            f.write(f'    {qpi[0]:.10f}  {qpi[1]:.10f}  {qpi[2]:.10f}' + ' 1\n')
    
    print(f'Saving matdyn.in file for calculation of phonon dispersion, this will output the phonon dispersion file {phonon_file_str}')
            
def plot_phonons(prefix, kpath_dict, axis=None, workdir='./phonons'):
    """
    Prepares input file for QE matdyn.x calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for SCF + PH calculations.
        kpath_dict (dict): dict generated by elphonpy.bands.get_simple_kpath , or modified to similar standard.
        
    Returns: 
        phonons_dataframe (pandas.DataFrame): The data which the phonons are plotted.
    """
    phonons_df = pd.read_csv(f'{workdir}/{str.lower(prefix)}.freq.gp', delim_whitespace=True, header=None)
    col_names = ['recip']

    cm_to_meV = 1 / 8.065610

    rng = np.arange(1, int(len(list(phonons_df))))
    for i in rng:
        col_names.append(f'Mode_{i}')

    phonons_df = pd.read_csv(f'{workdir}/{str.lower(prefix)}.freq.gp', names=col_names, delim_whitespace=True, header=None)

    # Create axis if none is supplied
    if axis == None:
        fig, axis = plt.subplots(figsize=[4,3], dpi=300)
        chose_ax = False
    # Chose an axis, don't create a new fig, ax
    else:
        chose_ax = True    
    
    mini = np.min(phonons_df.values * cm_to_meV)
    maxi = np.max(phonons_df.values * cm_to_meV)
    if mini > -5:
        miny = -5
    else:
        miny = mini+(0.2*mini)
    maxy = maxi + (0.2*maxi)
    
    axis.axhline(0, xmin=0, xmax=max(phonons_df['recip']), c='k', ls='--', lw=0.5, alpha=0.5)

    if isinstance(kpath_dict['path_symbols'][0], list):
        for i, high_sym in enumerate(join_last_to_first_latex(kpath_dict['path_symbols'])):
            sym_idx = kpath_dict['path_idx_wrt_kpt'][i]
            x_sym = phonons_df['recip'].iloc[sym_idx]
            axis.vlines(x_sym, ymin=miny, ymax=maxy+100, lw=0.3, colors='k')
            axis.text(x_sym/max(phonons_df['recip']), -0.05, f'{high_sym}', ha='center', va='center', transform=axis.transAxes) 

    else:
        for i, high_sym in enumerate(kpath_dict['path_symbols']):
            sym_idx = kpath_dict['path_idx_wrt_kpt'][i]
            x_sym = phonons_df['recip'].iloc[sym_idx]
            axis.vlines(x_sym, ymin=miny, ymax=maxy+100, lw=0.3, colors='k')
            axis.text(x_sym/max(phonons_df['recip']), -0.05, f'{high_sym}', ha='center', va='center', transform=axis.transAxes)

    for i in rng:
        axis.plot(phonons_df['recip'].values, phonons_df[f'Mode_{i}'].values*cm_to_meV, c='b', lw=1)
    
    axis.set_ylim(miny,maxy)
    axis.set_xlim(0,max(phonons_df['recip']))
    axis.set_xticks([])
    
    if chose_ax == False:
        axis.set_ylabel('Energy [meV]')
        fig.tight_layout()
        fig.savefig(f'{workdir}/{prefix}_phonons.png')
    
        return phonons_df

    if chose_ax == True:
        return axis, phonons_df
