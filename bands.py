


def bands_input_gen(prefix, structure, pseudo_dict, param_dict_scf, param_dict_bands, multE=1.3, band_points=50, workdir='./bands', copy_pseudo=False):
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
    
    high_sym_dict = get_simple_kpath(structure)

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