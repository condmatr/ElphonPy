import re
import subprocess
import pandas as pd
from elphonpy.pw import PWInput


def get_simple_kpath(structure, line_density=100):
    """
    Creates kpath for desired structure using SeeKPath, outputs kpath dictionary. 

    Args:
        structure (Pymatgen Structure or IStructure): Input structure.
        line_density (float): density of points along path per reciprocal unit distance.
        
    Returns:
        high_sym_dict (dict): Dictionary containing kpath information.
    """
    from pymatgen.symmetry.kpath import KPathSeek as seekpath
    import numpy as np
    skp = seekpath(structure, symprec=0.01)
    kp = skp.get_kpoints(line_density=line_density, coords_are_cartesian=False)

    path = skp.kpath
    kp_recip_list = list(path['kpoints'].values())

    kp_arrays = kp[0]
    kp_high_sym_list = kp[1]

    high_sym_dummy = []
    high_sym_idx = []
    high_sym_symbol = []
    high_sym_kpt = []

    kpt_out = []

    for i in range(len(kp_arrays)):
        if kp_high_sym_list[i] != '' and kp_high_sym_list[i] != kp_high_sym_list[i-1]:
            if i-1 in high_sym_dummy:
                break
            high_sym_dummy.append(i)
            if kp_high_sym_list[i] == 'GAMMA':
                high_sym_symbol.append('$\Gamma$')
                high_sym_kpt.append(kp_arrays[i].tolist())
            else:
                high_sym_symbol.append(kp_high_sym_list[i]) 
                high_sym_kpt.append(kp_arrays[i].tolist())

        if not np.array_equal(kp_arrays[i], kp_arrays[i-1]):
            kpt_out.append(kp_arrays[i].tolist())
        else:
            continue

    for i in range(len(kpt_out)):
        for j in kp_recip_list:
            if np.allclose(np.array(kpt_out[i]), np.array(j), rtol=1e-05,atol=1e-08):
                high_sym_idx.append(i)
                
    
    high_sym_dict = {'path_symbols':high_sym_symbol,
                     'path_kpoints':high_sym_kpt,
                     'path_idx_wrt_kpt':high_sym_idx,
                     'kpoints':kpt_out}
    
    return high_sym_dict


def bands_input_gen(prefix, structure, pseudo_dict, param_dict_scf, param_dict_bands, multE=1.5, line_density=100, workdir='./bands', copy_pseudo=False):
    """
    Prepares input files for QE Bands calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for SCF + NSCF bands calculations.
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use. Default to None.
        param_dict_scf (dict): A dict containing sections for SCF calculation input file ('system','control','electrons','kpoint_grid').
        param_dict_bands  (dict): A dict containing sections for NSCF bands calculation input file ('system','control','electrons').
        multE (float): Multiplier for pseudopotentials ecutwfc, ecutrho if not specified in scf_param_dict.
        line_density (float): density of points along path per reciprocal unit distance.
        workdir (str): target directory for writing SCF + NSCF bands input files. (default: './phonons').
        copy_pseudo (bool): Whether to copy pseudopotentials to current working directory in folder "pseudo". (default: False).
    
    Returns:
        high_sym_dict (dict): Dictionary containing kpath information.
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

    if 'ecutwfc' not in pmd_bands['system'].keys():
        pmd_bands['system'].update({'ecutwfc':min_ecutwfc*multE})
    if 'ecutrho' not in pmd_bands['system'].keys():
        pmd_bands['system'].update({'ecutrho':min_ecutrho*multE})
    
    high_sym_dict = get_simple_kpath(structure=structure, line_density=line_density)

    kp_out = high_sym_dict['kpoints']

    scf_calc = PWInput(structure=structure, pseudo=pseudopotentials, control=pmd_scf['control'],
                       electrons=pmd_scf['electrons'], system=pmd_scf['system'], cell=None,
                       kpoints_grid=pmd_scf['kpoint_grid'], ions=None)
    scf_calc.write_file(f'{workdir}/{prefix}_scf.in')

    bands_calc =  PWInput(structure=structure, pseudo=pseudopotentials,
                     control=pmd_bands['control'], electrons=pmd_bands['electrons'],
                     system=pmd_bands['system'])
    bands_calc.write_file('bands.temp')
    
    with open('bands.temp', 'r+') as f:
        temp = f.readlines()[:-2]
    f.close()
    
    with open(f'{workdir}/{prefix}_bands.in', 'w+') as f:
        for i in temp:
            f.write(i)
        f.write('K_POINTS crystal\n')
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
    
    return high_sym_dict

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
    
def plot_bands(prefix, structure, filband, fermi_e, y_min=None, y_max=None, savefig=True, save_dir='./bands'):
    """
    Plots electronic band structure from y_min to y_max.

    Args:
        prefix (str): prefix of output files for NSCF bands calculations
        structure (Pymatgen Structure or IStructure): Input structure.
        filband (str): Path to directory and filename for filband file output from bands.x calculation.
        fermi_e (float): Fermi energy in eV.
        y_min (float): minimum energy to plot. (optional, default=bands_min)
        y_max (float): maximum energy to plot. (optional, default=bands_max)
        savefig (bool): Whether or not to save fig as png.
        savedir (str): path to save directory if savefig == True. (default=./bands)

    Returns:
        bands_df (pandas DataFrame): Dataframe containing parsed band data.
    """
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=[4,3], dpi=300)
    high_sym_dict = get_simple_kpath(structure)
    bands_df, nbnds, kinfo = parse_filband(filband, npl=10, save_dir=save_dir)
    for i, high_sym in enumerate(high_sym_dict['path_symbols']):
        sym_idx = high_sym_dict['path_idx_wrt_kpt'][i]
        x_sym = bands_df['recip'].iloc[sym_idx]
        ax.vlines(x_sym, ymin=y_min, ymax=y_max, lw=0.3, colors='k')
        ax.text(x_sym/max(bands_df['recip']), -0.05, f'{high_sym}', ha='center', va='center', transform=ax.transAxes)

    ax.axhline(0, xmin=0, xmax=max(bands_df['recip']), c='k', ls='--', lw=0.5, alpha=0.5)

    for idx in range(1,len(bands_df.columns)-1):
        ax.plot(bands_df['recip'], bands_df[f'{idx}'].values - fermi_e, lw=1, c='b')
        
    if y_min != None and y_max != None:
        ax.set_ylim(y_min,y_max)
    
    ax.set_xlim(0,max(bands_df['recip']))
    ax.xaxis.set_visible(False)
    ax.set_ylabel('E-E$_{f}$ [eV]')
    if savefig = True:
        plt.savefig(f'{save_dir}/{prefix}_bands.png')
    
    return bands_df

def wannier_windows_info(band_df, save_dir='./bands'):
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
        for band in range(0, len(bands_df.columns)-1):
            band_min, band_max = np.min(bands_df[f'{band}'].values), np.max(bands_df[f'{band}'].values) 
            f.write(f'band # {band+1} :: min (eV) {band_min:.5f}, max (eV) {band_max:.5f}\n')
    f.close()