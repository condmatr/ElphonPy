import os
import pandas as pd
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
    j = 1
    
    if wannier_plot == True:

        j += 2
        wdata_list.append("wdata(1) = 'bands_plot = .TRUE.'")
        wdata_list.append("wdata(2) = 'begin kpoint_path'")
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

def epw_input_gen(prefix, structure, pseudo_dict, param_dict_scf, param_dict_nscf, param_dict_epw, kpath_dict,
                  wannier_plot=True, multE=1.0, workdir='./epw', copy_pseudo=False, coarse_only=False):
    """
    Prepares input file for EPW calculation, writes input file to workdir. 

    Args:
        prefix (str): prefix of input/output files for scf calculations.
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use. Default to None.
        param_dict_scf (dict): A dict containing sections for SCF input file ('system','control','electrons','kpoint_grid')
        param_dict_nscf (dict): A dict containing sections for NSCF input file ('system','control','electrons','kpoint_grid')
        param_dict_epw (dict): A dict containing sections for EPW input file ('inputepw','wannier_data','kq_grids')
        kpath_dict (dict): dict generated by elphonpy.bands.get_simple_kpath , 
or modified to similar standard.
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

    if coarse_only == True:
        param_dict_epw['inputepw'].update({'elph':True,
                                           'epwwrite':True,
                                           'epwread':False})
        with open(f'{workdir}/{prefix}_epw_wann_coarse.in', 'w+') as f:
            f.write('&inputepw\n')
            for item in param_dict_epw['inputepw'].items():
                f.write(f'  {item[0]} = {to_str(item[1])}' + '\n')
            f.write('\n')
            for item in wdata:
                f.write(f'  {item}' + '\n')
            f.write('\n')
            for i, nk in enumerate(param_dict_epw['kq_grids']['k_coarse']):
                f.write(f'  nk{i+1} = {nk}' + '\n')
            for i, nq in enumerate(param_dict_epw['kq_grids']['q_coarse']):
                f.write(f'  nq{i+1} = {nq}' + '\n')
            for i, nkf in enumerate(param_dict_epw['kq_grids']['k_fine']):
                f.write(f'  nkf{i+1} = {nkf}' + '\n')
            for i, nqf in enumerate(param_dict_epw['kq_grids']['q_fine']):
                f.write(f'  nqf{i+1} = {nqf}' + '\n')
            f.write('/\n')
        f.close()

    else:
        param_dict_epw['inputepw'].update({'elph':True,
                                           'epwwrite':True,
                                           'epwread':False})
        with open(f'{workdir}/{prefix}_epw_wann_coarse.in', 'w+') as f:
            f.write('&inputepw\n')
            for item in param_dict_epw['inputepw'].items():
                f.write(f'  {item[0]} = {to_str(item[1])}' + '\n')
            f.write('\n')
            for item in wdata:
                f.write(f'  {item}' + '\n')
            f.write('\n')
            for i, nk in enumerate(param_dict_epw['kq_grids']['k_coarse']):
                f.write(f'  nk{i+1} = {nk}' + '\n')
            for i, nq in enumerate(param_dict_epw['kq_grids']['q_coarse']):
                f.write(f'  nq{i+1} = {nq}' + '\n')
            for i, nkf in enumerate(param_dict_epw['kq_grids']['k_fine']):
                f.write(f'  nkf{i+1} = {nkf}' + '\n')
            for i, nqf in enumerate(param_dict_epw['kq_grids']['q_fine']):
                f.write(f'  nqf{i+1} = {nqf}' + '\n')
            f.write('/\n')
        f.close()
        param_dict_epw['inputepw'].update({'elph':True,
                                           'epwwrite':False,
                                           'epwread':True})
        with open(f'{workdir}/{prefix}_epw.in', 'w+') as f:
            f.write('&inputepw\n')
            for item in param_dict_epw['inputepw'].items():
                f.write(f'  {item[0]} = {to_str(item[1])}' + '\n')
            f.write('\n')
            for item in wdata:
                f.write(f'  {item}' + '\n')
            f.write('\n')
            for i, nk in enumerate(param_dict_epw['kq_grids']['k_coarse']):
                f.write(f'  nk{i+1} = {nk}' + '\n')
            for i, nq in enumerate(param_dict_epw['kq_grids']['q_coarse']):
                f.write(f'  nq{i+1} = {nq}' + '\n')
            for i, nkf in enumerate(param_dict_epw['kq_grids']['k_fine']):
                f.write(f'  nkf{i+1} = {nkf}' + '\n')
            for i, nqf in enumerate(param_dict_epw['kq_grids']['q_fine']):
                f.write(f'  nqf{i+1} = {nqf}' + '\n')
            f.write('/\n')
        f.close()

    
def plot_wannier_dft_bands(prefix, band_kpath_dict, fermi_e=0, reduce_wann=1, bands_dir='./bands', wann_dir='./epw', y_min=None, y_max=None, savefig=True, s=0.05):    
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
        savedir (str): path to save directory if savefig == True. (default=./epw)

    Returns:
        bands_df (pandas DataFrame): Dataframe containing parsed band data.
    """
    
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, ax = plt.subplots(figsize=[4,3], dpi=300)
    bands_df = pd.read_json(f'{bands_dir}/bands_reformatted.json')
    wann_bands_df = pd.read_csv(f'{wann_dir}/{prefix.lower()}_band.dat', delim_whitespace=True, names=['recip', 'band_data'])
    
    factor = bands_df['recip'].values[-1]/wann_bands_df['recip'].values[-1]
    
    y_min_bands = min(bands_df.values[:,1])
    y_max_bands = min(bands_df.values[:,-1])
    
    if y_min == None and y_max == None:
        y_min_wann = min(wann_bands_df['band_data'])-1
        y_max_wann = max(wann_bands_df['band_data'])+1
    
    else:
        y_min_wann = y_min + fermi_e
        y_max_wann = y_max + fermi_e
    
    for i, high_sym in enumerate(band_kpath_dict['path_symbols']):
        sym_idx = band_kpath_dict['path_idx_wrt_kpt'][i]
        x_sym = bands_df['recip'].iloc[sym_idx]
        ax.vlines(x_sym, ymin=y_min_bands, ymax=y_max_bands, lw=0.3, colors='k')
        ax.text(x_sym/max(bands_df['recip']), -0.05, f'{high_sym}', ha='center', va='center', transform=ax.transAxes)

    ax.axhline(fermi_e, xmin=0, xmax=max(bands_df['recip']), c='k', ls='--', lw=0.5, alpha=0.5)
    
    wann_bands_df = wann_bands_df.iloc[::reduce_wann, :]
    
    for idx in range(1,len(bands_df.columns)-1):
        ax.plot(bands_df['recip'], bands_df[f'{idx}'].values, lw=1, c='r', zorder=1)
    ax.scatter(wann_bands_df['recip']*factor, wann_bands_df['band_data'], s=s, c='k', zorder=2)
    
    ax.set_xlim(0,max(bands_df['recip']))
    ax.set_ylim(y_min_wann, y_max_wann)
    ax.xaxis.set_visible(False)
    ax.set_ylabel('Energy [eV]')
    fig.tight_layout()
    if savefig == True:
        plt.savefig(f'{wann_dir}/{prefix}_DFT_wann_bands.png')

def read_a2f(filename, print_info=True):
    """
    Reads a2F (or a2F_tr) file output from EPW, returns dataframe object, and prints calculation parameter info.

    Args:
        filename (str): Path to directory and filename for a2f file output from epw.x calculation.
        print_info (bool): Whether or not to print electronic smearing, fermi window, and summed EPC.

    Returns:
        a2f_df (pandas DataFrame): Dataframe containing parsed a2F data.
    """
    import pandas as pd
    phonon_smearings = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                tokens = line.split()[1:]
                phonon_smearings = [float(x) for x in tokens]
            if line.startswith('Electron'):
                el_smear = float(line.split()[-1])
            if line.startswith('Fermi'):
                fermi_win = float(line.split()[-1])
            if line.startswith('Summed'):
                sum_elph = float(line.split()[-1])
    
    if print_info == True:
        print(f'a2F FILE INFO: {filename} :\nElectronic smearing (eV) = {el_smear:.4f},\nFermi Window (eV) = {fermi_win:.4f},\nSummed EPC = {sum_elph:.3f}')

    a2f_df = pd.read_csv(filename, skipfooter=7, delim_whitespace=True, names=['energy'] + [f'phsmear_{ph}' for ph in phonon_smearings])

    return a2f_df

def allen_dynes(freq, a2f, mu=None):
    import numpy as np
    from scipy.integrate import trapezoid

    lambda_values = []
    
    def lambda_(freq, a2f):
        return trapezoid(y=2*(a2f/freq), x=freq)

    def omega_log(freq, a2f, lamb):
        return np.exp((2/lamb)*trapezoid(y=(a2f/freq)*np.log(freq), x=freq))

    for i in range(len(freq)):
        a2f_vals = a2f[:i] 
        omega_vals = freq[:i]

        lambda_values.append(trapezoid(y=2*(a2f_vals/omega_vals), x=omega_vals))

    lamb = lambda_(freq,a2f)
    ol = omega_log(freq,a2f,lamb)
    if mu != None:
        Tc = (ol*11.6/1.2) * np.exp((-1.04*(1+lamb))/(lamb-(mu*(1+0.62*lamb))))
    else:
        Tc = []
        for mu in np.arange(0.08,0.21,0.01):
            tc = (ol*11.6/1.2) * np.exp((-1.04*(1+lamb))/(lamb-(mu*(1+0.62*lamb)))) 
            Tc.append(tc)
        print('mu* = ', mu, 'Tc = ', Tc, ' K') 

    return ol, lamb, lambda_values, Tc


def get_degaussw_degaussq(files):
    degaussw = [] # empty list for electronic smearings
    degaussq = [] # empty list for phonon smearings
    print(files)
    for j, file_path in enumerate(files): # for each file passed to the function
        # read file
        with open(file_path, 'r') as file: 
            lines = file.readlines()
            for i, line in enumerate(lines): # line by line search
                # search for keywords and split strings
                if "Phonon smearing (meV)" in line and j == 0: # only get degaussq once (assuming the same for all files in dir)
                    try:
                        values = [float(val) for val in lines[i + 1].split()[1:]] # line starts with hash sym
                        degaussq.extend(values)
                    except ValueError:
                        raise ValueError
                if "Electron smearing (eV)" in line: # get degaussw for each file in list
                    try:
                        value = float(line.split()[-1])
                        degaussw.append(value)
                    except ValueError:
                        raise ValueError
            # close file
            file.close()
    # return smearing lists 
    return degaussw, degaussq
    

def plot_a2f_file(prefix, filename, degaussw, degaussq_list, dim_a2f=[2,5],
                  savefig=True, savedir=None, title=None):
    
    import matplotlib.pyplot as plt
    
    if dim_a2f == None: # Rows and columns of figure, 10 q smearings in file by default
        print('Provide dimensions of subplot array for a2f')

    # set column names of dataframe and make dataframe
    col_names = ['freq'] + [f'phsmear_{x:.3f}' for x in degaussq_list]
    a2f_df = pd.read_csv(filename, delim_whitespace=True, names=col_names, skipfooter=7, engine='python')

    # Plotting
    fig,axes = plt.subplots(dim_a2f[0],dim_a2f[1], figsize=[dim_a2f[1]*5, dim_a2f[0]*5], dpi=300, sharey=True, sharex=True)
    l = 0 # dummy counter row
    m = 0 # dummy counter column
    for i, dq in enumerate(col_names[1:]): # for each a2f column in a2f_df
        if m > dim_a2f[1]-1: # if row counter > # of rows
            l += 1           # increase row
            m = 0            # set column to 0

        ax = axes[l,m] # select axis for plotting

        ol, lamb, lambda_values, tcs = allen_dynes(a2f_df['freq'], a2f_df[dq])
        ax.plot(a2f_df['freq'], a2f_df[dq]) # plot a2f for column
        ax.plot(a2f_df['freq'], lambda_values) # plot lambda for column 
        
        # Setting title for each axis, xlimits, ylimits for each axis based on maximum values in a2f file
        ax.set_title(f'{degaussq_list[i]:.3f} meV', fontsize=14)
        ax.set_xlim(0,max(a2f_df['freq']))
        ax.set_ylim(0,max(a2f_df[col_names[1]]))

        # Only put labels on the left most (ylabel), or bottom most (xlabel) subplot axes
        if l == dim_a2f[0]-1: 
            ax.set_xlabel('Energy (meV)', fontsize=14)
            ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=12)
        if m == 0:
            ax.set_ylabel('$\\alpha^{2}F(\omega)$', fontsize=14)
            ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=12)
        m += 1

    # Prepare figure for saving and save
    if title == None:
        fig.suptitle(f'{prefix} e-smearing {degaussw:.2f}', y=1.07, fontsize=16)
    else:
        fig.suptitle(title, y=0.98, fontsize=18)
    fig.tight_layout()
    if savefig == True:
        if savedir == None:
            fig.savefig(f'./{prefix}.a2f_{degaussw:.2f}.png')
        else:
            fig.savefig(f'{savedir}/{prefix}.a2f_{degaussw:.2f}.png')

    return a2f_df

def plot_epw_smearing_convergence(prefix, files, degaussw_list, degaussq_list, savefig=True, savedir=None):
    import matplotlib.pyplot as plt
    col_names = ['freq'] + [f'phsmear_{dq:.3f}' for dq in degaussq_list] # set column names for df
    fig, ax = plt.subplots(1,2, figsize=[10,5]) # instantiate figure
    for dq in degaussq_list:
        omg_log = [] # empty list for omega_logs 
        lambdas = [] # empty list for lambdas
        for idx, file in enumerate(files):
            
            # Open file with pandas
            df = pd.read_csv(file, delim_whitespace=True, names=col_names, skipfooter=7, engine='python')
        
            # Integrate a2F(omega) for lambda, omega_log
            ol, lamb, lambda_values, tcs = allen_dynes(df['freq'], df[f'phsmear_{dq:.3f}'])
            lambdas.append(lamb)
            omg_log.append(ol)
        
        # plot omega_log for each phsmear, each degaussw
        ax[0].plot(degaussw_list, omg_log, label=f'{dq:.2f} meV') 
        # plot lambda for each phsmear, each degaussw
        ax[1].plot(degaussw_list, lambdas, label=f'{dq:.2f} meV') 

    # set ax labels, tiddy up figure, save figure
    ax[0].set_xlabel('Electronic Smearing (eV)') 
    ax[1].set_xlabel('Electronic Smearing (eV)')
    ax[0].set_ylabel('$\omega_{log}$ (meV)')
    ax[1].set_ylabel('$\lambda$')
    ax[1].legend()
    fig.tight_layout()
    if savefig == True:
        if savedir == None:
            fig.savefig(f'./{prefix}.a2f_convergence_smearing.png')
        else:
            fig.savefig(f'{savedir}/{prefix}.a2f_convergence_smearing.png')

def plot_epw_convergence(prefix, workdir, plot_a2f=True, plot_smear=True,
                         title='$\\alpha^{2}F(\omega)$ convergence: degaussq',
                         savefig=True, savedir=None, a2f_smear_idx=1, dim_a2f=[2,5]):
    import warnings
    import glob
    # Setup to ignore specific warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Construct the pattern for the file search
    search_pattern = f"{prefix}.a2f.*"
    full_path_pattern = os.path.join(workdir, search_pattern)
    matching_files = sorted(glob.glob(full_path_pattern))

    degaussw_list, degaussq_list = get_degaussw_degaussq(matching_files)

    if plot_a2f == True:
        a2f_df = plot_a2f_file(prefix=prefix, filename=matching_files[a2f_smear_idx], degaussw=degaussw_list[a2f_smear_idx],
                               degaussq_list=degaussq_list, dim_a2f=dim_a2f, savefig=savefig, savedir=workdir, title=title)

    if plot_smear == True:
        plot_epw_smearing_convergence(prefix=prefix, files=matching_files, degaussw_list=degaussw_list,
                                      degaussq_list=degaussq_list, savefig=savefig, savedir=workdir)
        
    return a2f_df
