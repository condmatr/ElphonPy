import os 

qe_pseudo_dir='/home/adam/qe-7.0/pslibrary.1.0.0/pbe/PSEUDOPOTENTIALS'

def get_pseudos(structure, pseudo_dict, copy_pseudo=True):
    
    """
    Retrieves pseudopotentials for calculations; returns dict of pseudopotential file names, min_ecutwfc, min_ecutrho for calculations

    Args:
        structure (Pymatgen Structure or IStructure): Input structure.
        pseudo_dict (dict): A dict of the pseudopotentials to use. Default to None.
        copy_pseudo (bool): Whether to copy pseudopotentials to current working directory in folder "pseudo".
        
    Returns:
        pseudopotentials (dict): dictionary of pseudopotentials for input structure.
        min_ecutwfc (float): minimum ecutwfc recommended for input structure with given pseudopotentials in Ry.
        min_ecutrho (float): minimum ecutrho recommended for input structure with given pseudopotentials in Ry.
    """
    
    if copy_pseudo == True:
        try:
            os.mkdir('./pseudo')
        except OSError as error:
            print(error)
        
    atom_list = list(structure.symbol_set)
    
    pseudopotentials = {}
    max_ecutwfc = 0
    max_ecutrho = 0
    for atom in atom_list:
        pseudo = pseudo_dict[atom]['pseudo']
        pseudopotentials.update({f'{atom}':pseudo_dict[atom]['pseudo']})
        
        if int(pseudo_dict[atom]['ecutwfc [Ry]']) > max_ecutwfc:
            min_ecutwfc = int(pseudo_dict[atom]['ecutwfc [Ry]'])
        if int(pseudo_dict[atom]['ecutrho [Ry]']) > max_ecutrho:
            min_ecutrho = int(pseudo_dict[atom]['ecutrho [Ry]'])
        
        if copy_pseudo == True:
            shutil.copyfile(f'{qe_pseudo_dir}/{pseudo}',
                        f'./pseudo/{pseudo}')
        else:
            continue
    
    if copy_pseudo == True:
        print('Copied pseudopotentials to ./pseudo')
    
    return pseudopotentials, min_ecutwfc, min_ecutrho