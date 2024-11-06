import re
import os
import subprocess
import pandas as pd
import numpy as np
import json
from elphonpy.pw import PWInput, get_ibrav_celldm
from elphonpy.pseudo import get_pseudos


def distance_kpt_spacing(high_sym_kpoints, line_density):
    kpath = []
    sym_kpt_idx = []
    idx = 0
    for i in range(len(high_sym_kpoints)-1):
        if high_sym_kpoints[i] != 'break' and high_sym_kpoints[i+1] != 'break':
            num_pts = np.int64(np.round(line_density*np.linalg.norm(np.array(high_sym_kpoints[i]) - np.array(high_sym_kpoints[i+1])),0))
            kpoints = np.linspace(high_sym_kpoints[i], high_sym_kpoints[i+1], num_pts)
            if i == 0:
                kpath = kpath + kpoints.tolist()
                sym_kpt_idx.append(idx)
                idx += num_pts - 1
                sym_kpt_idx.append(idx)
            else:
                kpath = kpath + kpoints[1:].tolist()
                idx += num_pts - 1
                sym_kpt_idx.append(idx)
        else:
            continue

    return kpath, sym_kpt_idx

def get_simple_kpath(structure, line_density=100, filpath='./kpath.json'):
    """
    Creates kpath for desired structure using SeeKPath, outputs kpath dictionary.

    Args:
	structure (Pymatgen Structure or IStructure): Input structure.
        line_density (float): density of points along path per reciprocal unit distance.

    Returns:
	kpath_dict (dict): Dictionary containing kpath information.
    """
    from pymatgen.symmetry.kpath import KPathSeek as seekpath
    import numpy as np
    import json
    skp = seekpath(structure, symprec=0.01)
    kpath = skp.kpath
    sym_kpt_list_clean = []
    sym_kpt_list = []

    for i, sym_path in enumerate(kpath['path']):
        for j, sym in enumerate(sym_path):
            if j == len(sym_path)-1 and i != len(kpath['path'])-1:
                sym_kpt_list.append(kpath['kpoints'][sym])
                sym_kpt_list.append('break')
                sym_kpt_list_clean.append(kpath['kpoints'][sym])
            else:
                sym_kpt_list.append(kpath['kpoints'][sym])
                sym_kpt_list_clean.append(kpath['kpoints'][sym])
            
    kpoint_list, sym_idx = distance_kpt_spacing(sym_kpt_list, line_density)
    sym_list = kpath['path']

    for i in range(len(sym_list)):
        if sym_list[i] == 'GAMMA':
            sym_list[i] = '$\Gamma$'

    kpath_dict = {'path_symbols':sym_list,
                  'path_kpoints':sym_kpt_list_clean,
                  'path_idx_wrt_kpt':sym_idx,
                  'kpoints':kpoint_list}

    with open(filpath, 'w') as f:
        json.dump(kpath_dict, f, default=int)

    return kpath_dict


# def get_custom_kpath(symbol_kpoints_dict, line_density=100):
#     """
#     Creates kpath for desired structure using custom kpath, outputs kpath dictionary. 

#     Args:
#         symbol_kpoints_dict (dict): A dictionary containing 'path_symbols':a list of symbol strings.
#                                                             'path_kpoints':a list of path kpoints.  
#         e.g.
#         symbol_kpoints_dict = {'path_symbols':['W','L','$\Gamma$','X','W','K'],
#                                'path_kpoints':[[0.500, 0.250, 0.750],
#                                                [0.500, 0.500, 0.500],
#                                                [0.000, 0.000, 0.000],
#                                                [0.500, 0.000, 0.500],
#                                                [0.500, 0.250, 0.750],
#                                                [0.625, 0.250, 0.625]
#                                               ]}
#         line_density (float): density of points along path per reciprocal unit distance.
        
#     Returns:
#         kpath_dict (dict): Dictionary containing kpath information.
#     """
    
#     sym_list = symbol_kpoints_dict['path_symbols']
#     sym_kpt_list = symbol_kpoints_dict['path_kpoints']
#     high_sym_kpoints = np.array(sym_kpt_list)
        
#     kpoint_list, sym_idx = distance_kpt_spacing(high_sym_kpoints, line_density)
    
#     for i in range(len(sym_list)):
#         if sym_list[i] == 'GAMMA':
#             sym_list[i] = '$\Gamma$'
        
#     kpath_dict = {'path_symbols':sym_list,
#                   'path_kpoints':sym_kpt_list,
#                   'path_idx_wrt_kpt':sym_idx,
#                   'kpoints':kpoint_list}
    
#     return kpath_dict

def get_custom_kpath(structure, symbol_kpoints_dict, line_density=100, filpath='./kpath.json'):
    """
    Creates kpath for desired structure using SeeKPath, outputs kpath dictionary. 

    Args:
        structure (Pymatgen Structure or IStructure): Input structure.
        symbol_kpoints_dict (dict): A dictionary containing 'path_symbols':a list of symbol strings.
                                                            'path_kpoints':a list of path kpoints.  
        e.g.
        symbol_kpoints_dict = {'path_symbols':['W','L','$\Gamma$','X','W','K'],
                               'path_kpoints':[[0.500, 0.250, 0.750],
                                               [0.500, 0.500, 0.500],
                                               [0.000, 0.000, 0.000],
                                               [0.500, 0.000, 0.500],
                                               [0.500, 0.250, 0.750],
                                               [0.625, 0.250, 0.625]
                                              ]}
        line_density (float): density of points along path per reciprocal unit distance.
        
    Returns:
        kpath_dict (dict): Dictionary containing kpath information.
    """
    import numpy as np
    
    recip_lat = structure.lattice.reciprocal_lattice

    kpoints = np.array(symbol_kpoints_dict['path_kpoints'])

    all_kp = []

    for i in range(len(kpoints)-1):

        kpi = kpoints[i]
        kpf = kpoints[i+1]

        weight = round(recip_lat.get_all_distances(kpi, kpf)[0][0]*line_density)

        kp_section = np.concatenate((np.linspace(kpi[0], kpf[0], weight).reshape(-1,1), np.linspace(kpi[1],kpf[1], weight).reshape(-1,1), np.linspace(kpi[2],kpf[2], weight).reshape(-1,1)), axis=1)
        all_kp.append(kp_section)

    kp_arrays = np.vstack(all_kp)
    
    kp_sym = symbol_kpoints_dict['path_symbols']
    path_kp = symbol_kpoints_dict['path_kpoints']
    kp_sym_idx = []
    kpt_out = []

    j = 0
    for i in range(len(kp_arrays)-1):

        kpt = kp_arrays[i]

        if not np.allclose(kpt, kp_arrays[i+1], rtol=1e-08,atol=1e-08):
            kpt_out.append(kpt.tolist())
            # print(True)
        if np.allclose(kpt, path_kp[j], rtol=1e-08,atol=1e-08):
            # print(kpt)
            if i == 0:
                kp_sym_idx.append(0)
            else:
                kp_sym_idx.append(i-j+1)
            j+=1

    kp_sym_idx.append(len(kp_arrays)-j)
    kpt_out.append(kp_arrays[-1].tolist())

    kpath_dict = {'path_symbols':kp_sym,
                  'path_kpoints':path_kp,
                  'path_idx_wrt_kpt':kp_sym_idx,
                  'kpoints':kpt_out}
    with open(filpath, 'w') as f:
        json.dump(kpath_dict, f, default=int)
    return kpath_dict


def bands_input_gen(prefix, structure, pseudo_dict, param_dict_scf, param_dict_bands, kpath_dict, multE=1.0, rhoe=None, workdir='./bands', copy_pseudo=False):
    """
    Prepares input files for QE Bands calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for SCF + NSCF bands calculations.
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use. Default to None.
        param_dict_scf (dict): A dict containing sections for SCF calculation input file ('system','control','electrons','kpoint_grid').
        param_dict_bands (dict): A dict containing sections for NSCF bands calculation input file ('system','control','electrons').
        kpath_dict (dict): dict generated by elphonpy.bands.get_simple_kpath , or modified to similar standard.
        multE (float): Multiplier for pseudopotentials ecutwfc, ecutrho if not specified in scf_param_dict.
        line_density (float): density of points along path per reciprocal unit distance.
        workdir (str): target directory for writing SCF + NSCF bands input files. (default: './phonons').
        copy_pseudo (bool): Whether to copy pseudopotentials to current working directory in folder "pseudo". (default: False).
    """
    try:
        os.mkdir(workdir)
    except OSError as error:
        print(error)
        
    pmd_scf = param_dict_scf
    pmd_bands = param_dict_bands

    pseudopotentials, min_ecutwfc, min_ecutrho = get_pseudos(structure, pseudo_dict, copy_pseudo=copy_pseudo)
    
    if 'celldm(1)' and 'ibrav' not in pmd_scf['system'].keys():
        celldm_dict = get_ibrav_celldm(structure)
        pmd_scf['system'].update(celldm_dict)
        pmd_bands['system'].update(celldm_dict)
 
    if 'ecutwfc' not in pmd_scf['system'].keys():
        pmd_scf['system'].update({'ecutwfc':min_ecutwfc*multE})
    if 'ecutrho' not in pmd_scf['system'].keys():
        pmd_scf['system'].update({'ecutrho':min_ecutrho*multE})
        if rhoe != None:
            pmd['system'].update({'ecutrho':min_ecutwfc*multE*rhoe})

    if 'ecutwfc' not in pmd_bands['system'].keys():
        pmd_bands['system'].update({'ecutwfc':min_ecutwfc*multE})
    if 'ecutrho' not in pmd_bands['system'].keys():
        pmd_bands['system'].update({'ecutrho':min_ecutrho*multE})
        if rhoe != None:
            pmd['system'].update({'ecutrho':min_ecutwfc*multE*rhoe})

    kp_out = kpath_dict['kpoints']

    scf_calc = PWInput(structure=structure, pseudo=pseudopotentials, control=pmd_scf['control'],
                       electrons=pmd_scf['electrons'], system=pmd_scf['system'], cell=None,
                       kpoints_grid=pmd_scf['kpoint_grid'], ions=None)
    scf_calc.write_file(f'{workdir}/{prefix}_scf.in')

    bands_calc =  PWInput(structure=structure, pseudo=pseudopotentials,
                     control=pmd_bands['control'], electrons=pmd_bands['electrons'],
                     system=pmd_bands['system'])
    bands_calc.write_file('bands.temp')
    
    with open('bands.temp', 'r+') as f:
        temp = f.readlines()
        for i, line in enumerate(temp):
            if 'K_POINTS' in line:
                temp.remove(temp[i])
                temp.remove(temp[i])
    f.close()

    os.remove('bands.temp')
    
    with open(f'{workdir}/{prefix}_bands.in', 'w+') as f:
        for i in temp:
            f.write(i)
        f.write('\nK_POINTS crystal\n')
        f.write(f"{len(kp_out)}\n")
        for i in range(len(kp_out)):
            kpi = kp_out[i]
            f.write(f'    {kpi[0]:.8f}  {kpi[1]:.8f}  {kpi[2]:.8f}' + ' 1\n')

    f.close()

    with open(f'{workdir}/{prefix}_bandsx.in', 'w+') as f:
        f.write('&bands\n')
        f.write(f"  prefix = '{prefix.lower()}'\n  outdir = './'\n  filband = '{prefix}_band.dat'\n")
        f.write('/\n')
    f.close()

def get_fermi_e(filename):
    """
    Pulls Fermi Energy from specified file. 

    Args:
        filename (str): Path to directory and file name to get Fermi Energy from.
    
    Returns:
        fermi_energy (float): Fermi energy from calculation.
    """
    fermi = subprocess.run('grep Fermi' + f' {filename} ' + "| awk '{print $5}'", shell=True, capture_output=True)
    return float(fermi.stdout)

def parse_filband(filband, npl=10, save=True, save_dir='./bands'):
    """
    Parser for filband output from bands.x calculation. 

    Args:
        filband (str): Path to directory and filename for filband file output from bands.x calculation.
        npl (int): number per line (bands.x calculation: 10, matdyn.x calculation: 6).
        save (bool): Whether to save parsed data formatted for easy access later by pandas.
        save_dir (str): directory to save reformatted_bands.json file if save == True.
    Returns:
        bands_df (pandas DataFrame): parsed band data.
        nbnd (int): number of bands found in filband file.
        kinfo (list): list of kpoints from which reciprocal distance was calculated.
    """
    f=open(filband,'r')
    lines = f.readlines()

    header = lines[0].strip()
    line = header.strip('\n')
    shape = re.split('[,=/]', line)
    nbnd = int(shape[1])
    nks = int(shape[3])
    eig = np.zeros((nks, nbnd), dtype=np.float32)

    dividend = nbnd
    divisor = npl
    div = nbnd // npl + 1 if nbnd % npl == 0 else nbnd // npl + 2 
    kinfo=[]
    for index, value in enumerate(lines[1:]):
        value = value.strip(' \n')
        quotient = index // div
        remainder = index % div

        if remainder == 0:
            kinfo.append([float(x) for x in value.split()])
        else:
            value = re.split('[ ]+', value)
            a = (remainder - 1) * npl
            b = a + len(value)
            eig[quotient][a:b] = value
    f.close()
    
    recip_dis = 0
    recip = [0]
    for i in range(1, len(kinfo)):
        a = kinfo[i-1]
        b = kinfo[i]
        dis_a_b = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2)
        recip_dis += dis_a_b
        recip.append(recip_dis)
    
    recip = np.array(recip)
    col_names = ['recip']
    band_names = [str(x) for x in range(0,nbnd)]
    for name in band_names:
        col_names.append(name)
    print(col_names)
    bands_df = pd.DataFrame(np.hstack((recip.reshape(-1,1), eig)), columns=col_names)
    
    if save == True:
        bands_df.to_json(f'{save_dir}/bands_reformatted.json')
    
    return bands_df, nbnd, kinfo

def latexify(symbol):
    if '_' in symbol:
        parts = symbol.split('_')
        return f"{parts[0]}$_{{{parts[1]}}}$"
    if symbol == 'GAMMA':
        return '$\Gamma$'
    else:
        return symbol

def join_last_to_first_latex(lst):

    lst_latexified = [[latexify(symbol) for symbol in inner_list] for inner_list in lst]

    result = []
    for i in range(len(lst_latexified)):
        inner_list = lst_latexified[i]
        if i < len(lst_latexified) - 1:
            joined_element = inner_list[-1] + '|' + lst_latexified[i + 1][0]
            result.extend(inner_list[:-1] + [joined_element])
            lst_latexified[i + 1] = lst_latexified[i + 1][1:]
        else:
            result.extend(inner_list)
    return result

def plot_bands(prefix, filband, fermi_e, kpath_dict, axis=None, y_min=None, y_max=None, savefig=True, save_dir='./bands'):
    """
    Plots electronic band structure from y_min to y_maxis.

    Args:
        prefix (str): prefix of output files for NSCF bands calculations
        filband (str): Path to directory and filename for filband file output from bands.x calculation.
        kpath_dict (dict): dict generated by elphonpy.bands.get_simple_kpath , or modified to similar standard.
        fermi_e (float): Fermi energy in eV.
        y_min (float): minimum energy to plot wrt fermi energy. [eV] (optional, default=bands_min)
        y_max (float): maximum energy to plot wrt fermi energy. [eV] (optional, default=bands_max)
        savefig (bool): Whether or not to save fig as png.
        savedir (str): path to save directory if savefig == True. (default=./bands)

    Returns:
        bands_df (pandas DataFrame): Dataframe containing parsed band data.
    """
    # import matplotlib
    import matplotlib.pyplot as plt

    # pull in data from filband data file produced by bands.x
    bands_df, nbnds, kinfo = parse_filband(filband, npl=10, save_dir=save_dir)

    # Create axis if none is supplied
    if axis == None:
        fig, axis = plt.subplots(figsize=[4,3], dpi=300)
        chose_ax = False
    # Chose an axis, don't create a new fig, ax
    else:
        chose_ax = True

    # Set appropriate y_min if none is chosen
    if y_min == None:
        y_min = 1.05*min(bands_df['0'])
    if y_max == None:
        y_max = 1.05*max(bands_df[bands_df.columns[-1]])

    # Gotta do this for broken kpaths, not sure how to improve to make this more readable/predictable when modifying
    if isinstance(kpath_dict['path_kpoints'][0][0], list):
        for i, high_sym in enumerate(join_last_to_first_latex(kpath_dict['path_symbols'])):
            sym_idx = kpath_dict['path_idx_wrt_kpt'][i]
            x_sym = bands_df['recip'].iloc[sym_idx]
            axis.vlines(x_sym, ymin=y_min, ymax=y_max, lw=0.3, colors='k')
            axis.text(x_sym/max(bands_df['recip']), -0.05, f'{high_sym}', ha='center', va='center', transform=axis.transAxes)
    
    # Normal routine for continuous kpath
    else:
        for i, high_sym in enumerate(latexify(kpath_dict['path_symbols'])):
            sym_idx = kpath_dict['path_idx_wrt_kpt'][i]
            x_sym = bands_df['recip'].iloc[sym_idx]
            axis.vlines(x_sym, ymin=y_min*1.05, ymax=y_max*1.05, lw=0.3, colors='k')
            axis.text(x_sym/max(bands_df['recip']), -0.05, f'{high_sym}', ha='center', va='center', transform=axis.transAxes)

    # plot bands from bands_df, reference zero to the Fermi level supplied from fermi_e variable
    for idx in range(1,len(bands_df.columns)-1):
        axis.plot(bands_df['recip'], bands_df[f'{idx}'].values - fermi_e, lw=1, c='b')

    # plot a horizontal line denoting the Fermi level (0 eV)
    axis.axhline(0, ls='dashed', c='k', lw=0.5 )

    # set ax limits
    axis.set_ylim(y_min,y_max)

    # save the figure if required
    if savefig == True:
        fig.tight_layout()
        fig.savefig(f'{save_dir}/{prefix}_bands.png')

    # If an ax is not chose, return the fig, ax, bands_df
    if chose_ax==True:
        axis.set_xlim(0,max(bands_df['recip']))
        axis.xaxis.set_visible(False)
        
        return axis, bands_df
    # if not, return just the bands_df
    else:
        axis.set_xlim(0,max(bands_df['recip']))
        axis.xaxis.set_visible(False)
        axis.set_ylabel('E - E$_{F}$ [eV]')
        
        return bands_df

def wannier_windows_info(bands_df, fermi_e, save_dir='./bands'):
    """
    Saves a data file with fermi energy, and band minimum and band maximum for each band in band directory for ease of setting 
    wannier90 disentanglement windows in epw calculation step.
    
    Generates wannier_info.dat file in save_dir

    Args:
        band_df (bool): Whether or not to save fig as png.
        savedir (str): path to save directory.
    """
       
    with open(f'{save_dir}/wannier_info.dat', 'w+') as f:
        f.write(f'Fermi Energy (SCF): {fermi_e}\n')
        f.write(f'band#       min(eV)       max(eV)\n')
        for band in range(0, len(bands_df.columns)-1):
            band_min, band_max = np.min(bands_df[f'{band}'].values), np.max(bands_df[f'{band}'].values) 
            f.write(f'{band+1}      {band_min:.5f}      {band_max:.5f}\n')
    f.close()
