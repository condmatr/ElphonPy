import os
from elphonpy.pw import scf_input_gen, nscf_input_gen, to_str
from elphonpy.pseudo import get_pseudos

def epw_wdata(param_dict_epw, wannier_plot, kpath_dict):
    """
    Generates section of EPW input file directly passed to Wannier90 input file for wannierization.

    Args:
        param_dict_epw (dict): A dict containing sections for SCF calculation input file ('inputepw', 'wannier_data'  'kq_grids').
        wannier_plot (bool): Whether to plot Wannier Function representation of bands after wannierization. (default: True)
        kpath_dict (dict): dict generated by elphonpy.bands.get_simple_kpath , or modified to similar standard.

    Returns:
        wdata_list (list of strings): List containing strings for wdata section of EPW input file.
    """
    wann_dict = param_dict_epw['wannier_data']
    wdata_list = []
    j = 0
    
    if wannier_plot == True:

        j = 2
        wdata_list.append("wdata(0) = 'bands_plot  = .TRUE.'")
        wdata_list.append("wdata(1) = 'begin kpoint_path'")

        for i in range(len(kpath_dict['path_kpoints'])-1):

            kpt_1, kpt_sym_1 = kpath_dict['path_kpoints'][i], kpath_dict['path_symbols'][i]
            kpt_2, kpt_sym_2 = kpath_dict['path_kpoints'][i+1], kpath_dict['path_symbols'][i+1]

            if kpt_sym_1 == '$\Gamma$':
                kpt_sym_1 = 'G'
            if kpt_sym_2 == '$\Gamma$':
                kpt_sym_2 = 'G'

            wdata_list.append(f"wdata({j}) = '{kpt_sym_1} {kpt_1[0]} {kpt_1[1]} {kpt_1[2]} {kpt_sym_2} {kpt_2[0]} {kpt_2[1]} {kpt_2[2]}'")
            j += 1

        wdata_list.append(f"wdata({j}) = 'end kpoint_path'")
        wdata_list.append(f"wdata({j+1}) = 'bands_plot_format = gnuplot xmgrace'")
        j += 2
    
    for i, key in enumerate(wann_dict.keys()):
        wdata_list.append(f"wdata({i+j}) = '{key} = {to_str(wann_dict[key])}'")
        
    return wdata_list

def epw_input_gen(prefix, structure, pseudo_dict, param_dict_scf, param_dict_nscf, param_dict_epw, kpath_dict, wannier_plot=True, multE=1.0, workdir='./epw', copy_pseudo=False):
    """
    Prepares input file for EPW calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for scf calculations.
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use. Default to None.
        param_dict_scf (dict): A dict containing sections for SCF input file ('system','control','electrons','kpoint_grid')
        param_dict_nscf (dict): A dict containing sections for NSCF input file ('system','control','electrons','kpoint_grid')
        param_dict_epw (dict): A dict containing sections for EPW input file ('inputepw','wannier_data','kq_grids')
        kpath_dict (dict): dict generated by elphonpy.bands.get_simple_kpath , or modified to similar standard.
        wannier_plot (bool): Whether to plot Wannier Function representation of bands after wannierization. (default: True)
        multE (float): Multiplier for pseudopotentials ecutwfc, ecutrho if not specified in param_dict.
        workdir (str): target directory for writing SCF input file.
        copy_pseudo (bool): Whether to copy pseudopotentials to current working directory in folder "pseudo".
    """
    
    try:
        os.mkdir(f'{workdir}')
    except OSError as error:
        print(error)
        
    scf_input_gen(prefix, structure, pseudo_dict, param_dict_scf, multE=multE, workdir=workdir, copy_pseudo=copy_pseudo)
    nscf_input_gen(prefix, structure, pseudo_dict, param_dict_nscf, multE=multE, workdir=workdir, copy_pseudo=False)
    
    pseudopotentials, min_ecutwfc, min_ecutrho = get_pseudos(structure, pseudo_dict, copy_pseudo=copy_pseudo)
    
    wdata = epw_wdata(param_dict_epw, wannier_plot, kpath_dict)
    
    with open(f'{workdir}/{prefix}_epw.in', 'w+') as f:
        f.write('&inputepw\n')
        for item in param_dict_epw['inputepw'].items():
            f.write(f'  {item[0]}={to_str(item[1])}' + ',\n')
        f.write('\n')
        for item in wdata:
            f.write(f'  {item}' + ',\n')
        f.write('\n')
        for i, nk in enumerate(param_dict_epw['kq_grids']['k_coarse']):
            f.write(f'  nk{i+1} = {nk}' + ',\n')
        for i, nq in enumerate(param_dict_epw['kq_grids']['q_coarse']):
            f.write(f'  nq{i+1} = {nq}' + ',\n')
        for i, nkf in enumerate(param_dict_epw['kq_grids']['k_fine']):
            f.write(f'  nkf{i+1} = {nkf}' + ',\n')
        for i, nqf in enumerate(param_dict_epw['kq_grids']['q_fine']):
            f.write(f'  nqf{i+1} = {nqf}' + ',\n')
        f.write('/')
    f.close()
    
def plot_wannier_dft_bands(prefix, band_kpath_dict, fermi_e=0, bands_dir='./bands', wann_dir='./epw', savefig=True):    
    """
    Plots wannier tight-binding model band structure over top of DFT band structure for comparison.

    Args:
        prefix (str): prefix of output files for NSCF bands calculations
        filband (str): Path to directory and filename for filband file output from bands.x calculation.
        kpath_dict (dict): dict generated by elphonpy.bands.get_simple_kpath , or modified to similar standard.
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
    bands_df = pd.read_json(f'{bands_dir}/bands_reformatted.json')
    wann_bands_df = pd.read_csv(f'{wann_dir}/{prefix}_band.dat', delim_whitespace=True, names=['recip', 'band_data'])
    
    factor = bands_df['recip'].values[-1]/wann_bands_df['recip'].values[-1]
    y_min_wann = min(wann_bands_df['band_data'])-1
    y_max_wann = max(wann_bands_df['band_data'])+1
    y_min = min(bands_df.values[:,0])
    y_max = min(bands_df.values[:,-1])
    
    for i, high_sym in enumerate(kpath_dict['path_symbols']):
        sym_idx = kpath_dict['path_idx_wrt_kpt'][i]
        x_sym = bands_df['recip'].iloc[sym_idx]
        ax.vlines(x_sym, ymin=y_min, ymax=y_max, lw=0.3, colors='k')
        ax.text(x_sym/max(bands_df['recip']), -0.05, f'{high_sym}', ha='center', va='center', transform=ax.transAxes)

    ax.axhline(fermi_e, xmin=0, xmax=max(bands_df['recip']), c='k', ls='--', lw=0.5, alpha=0.5)

    for idx in range(1,len(bands_df.columns)-1):
        ax.plot(bands_df['recip'], bands_df[f'{idx}'].values, lw=1, c='r', zorder=1)
    ax.scatter(wann_bands_df['recip']*factor, wann_bands_df['band_data'], s=0.05, c='k', zorder=2)
    
    ax.set_xlim(0,max(bands_df['recip']))
    ax.set_ylim(y_min_wann, y_max_wann)
    ax.xaxis.set_visible(False)
    ax.set_ylabel('Energy [eV]')
    if savefig == True:
        plt.savefig(f'{wann_dir}/{prefix}_DFT_wann_bands.png')