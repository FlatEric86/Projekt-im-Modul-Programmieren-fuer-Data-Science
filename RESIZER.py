import copy as cp
from PIL import Image
import numpy as np
import sys, os
import matplotlib.pyplot as plt

data_pott = './DATEN_POTT_test'
N_i_px = 25
N_j_px = 25





def resize(arr):
    '''
    Das Schreiben eines eigenen Image-Array-Skalierers war um Weiten schneller, als
    sich mit den wenig gut gegebenen Alternativen rum zu ärgern.
    Die Funktion ist erstmal relativ hart auf das halbieren von Images gecodet, da
    dies meinem Zweck völlig genügt. 
    Die Basis dient die arithmetische Mittelung der nächsten Nachbarpixel. 
    Beim Halbieren sind dies genau 4, sofern die Dimensionierung des Urbild-Arrays
    in beiden Richtungen geraden Zahlen entspricht.
    Mehr dazu siehe Dokumentation.
    '''
    I = np.shape(arr)[0]
    J = np.shape(arr)[1]
    resized_arr = np.ones((I//2, J//2))
    for i_roof in range(I//2):
        for j_roof in range(J//2):
            resized_arr[i_roof, j_roof] = (1/4)*np.sum(arr[2*i_roof:2*i_roof + 2, 2*j_roof:2*j_roof + 2])

    return resized_arr




N = 0
for data_set in os.listdir(data_pott):
    for dir_name in os.listdir(os.path.join(data_pott, data_set)):
        if dir_name == 'output':
            for sol_dir_name in os.listdir(os.path.join(data_pott, data_set, dir_name)):    
                    for fname in os.listdir(os.path.join(data_pott, data_set, dir_name,sol_dir_name)):
                        if 'resized' in fname:
                            continue

                        try:
                            perm_arr = np.loadtxt(os.path.join(data_pott, data_set, dir_name,sol_dir_name,fname), delimiter=',')
                        except ValueError:
                            print(os.listdir(os.path.join(data_pott, data_set, dir_name,sol_dir_name, fname)))
                        # plt.imshow(perm_arr)
                        # plt.show()

                        # skaliere Image-datei neu
                        perm_arr = resize(perm_arr)

                        # plt.imshow(perm_arr)
                        # plt.show()

                        # schreibe skaliertes Image als CSV-Matrix raus

                        fname_conj = fname[:-4] + '_resized.csv'    

                        np.savetxt(os.path.join(data_pott, data_set, dir_name,sol_dir_name, fname_conj), perm_arr, delimiter=',')

                        N += 1

                        # if N == 10:
                        #     sys.exit()

                            
    
