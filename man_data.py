#manege the data

import defines
import pandas as pd     #read csv files
import numpy as np


def make_dataFrame(file):
    data_frame = pd.read_csv(file)
    data_frame = data_frame.rename(columns={'spacegroup' : 'sg',
                            'number_of_total_atoms' : 'Natoms',
                            'percent_atom_al' : 'x_Al',
                            'percent_atom_ga' : 'x_Ga',
                            'percent_atom_in' : 'x_In',
                            'lattice_vector_1_ang' : 'ang_a',
                            'lattice_vector_2_ang' : 'ang_b',
                            'lattice_vector_3_ang' : 'ang_c',
                            'lattice_angle_alpha_degree' : 'ang_alpha',
                            'lattice_angle_beta_degree' : 'ang_beta',
                            'lattice_angle_gamma_degree' : 'ang_gamma',
                            'formation_energy_ev_natom' : 'E_f',
                            'bandgap_energy_ev' : 'E_bg'})
    return data_frame


def put2grups(data_frame,class_vec_name, data_col_name):
    classification_vec = np.unique((data_frame[class_vec_name]).values)
    new_df=data_frame.loc[:, [class_vec_name, data_col_name]]
    by_class = new_df.groupby(class_vec_name)
    datasets = pd.DataFrame(columns=classification_vec)
    for groups, data in by_class:
        for ind in range(len(data[data_col_name].values)):
            datasets.loc[ind,groups]=(data[data_col_name].values)[ind]
    return datasets

def data2csv(data_frame):
    x=1;
