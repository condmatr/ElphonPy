def get_total_energy(filename):
    # Open the text file and read its contents
    with open(f'{filename}', 'r') as file:
        contents = file.read()

    # Find the line that contains the total energy value
    total_energy_line = [line for line in contents.split('\n') if line.startswith('!    total energy')][0]
    total_energy_str = total_energy_line.split('=')[1].strip()
    total_energy = float(total_energy_str.split()[0])
    
    return total_energy   

def conv_thresh(structure, conv_param, energies, thresh=1.0e-5):
    """
    Returns convergence parameter value given a list of energies and convergence params and a threshold [eV/atom]
    
    Args:
        structure (Pymatgen Structure or IStructure): Input structure.
        conv_params (list or array) : Convergence parameter under study.
        energies (list or array) : Calculated total energy corresponding to the convergence parameter.
        thresh (float) : Threshold for difference in total energy / number of atoms in input structure. 
    
    Returns:
        selected_param (individual conv_param type) : Returns selected convergence parameter given threshold.
        diff (float) : difference in energy between selected_param and the convergence parameter immediately before. [eV/atom] 
    """
    num_atoms = len(structure.sites)
    for i in range(1,len(energies)):
        if 13.6*abs(energies[i]-energies[i-1])/num_atoms <= thresh:
            selected_param = conv_param[i]
            diff = 13.6*abs(energies[i]-energies[i-1])/num_atoms
            break
    return selected_param, diff