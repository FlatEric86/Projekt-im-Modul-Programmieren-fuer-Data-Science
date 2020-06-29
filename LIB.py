import matplotlib.pyplot as plt 
import numpy as np
import random as rand
import os
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn

def compute_corner_points(non_perm_obj):
    '''
    Eine wenig elegante Funktion zum ermitteln der Eckpunkte (Koordinaten) nichtpermeabler Bereiche im 
    Lösungsgebiet, um sie in Form von Linienplots in die Animation plotten zu können.
    Die Funktion funktioniert nur auf ein! nichtpermeables Objekt, welches zudem rechteckig sein muss.
    Ansatz: 
    Beim rechtwinkligen Objekten müssen die Eckpunkte (für 2D - > N=4) die längste Distanz
    zum Volumen-/Flächenschwerpunkt haben. Flächenschwerpunkt ist bei diskreten äquidistant homogen 
    konstituierten Punktwolken der mittlere Ortsvektor aller Punkte.
    '''
    # extrahiere die Koordinaten des nichtpermeablen Bereichs (wenn Pixelwert == 0).
    non_perm_obj_coords = []

    transpose_non_perm_obj = non_perm_obj.T
    for i in range(np.shape(transpose_non_perm_obj)[0]):
        for j in range(np.shape(transpose_non_perm_obj)[1]):
            if transpose_non_perm_obj[i,j] == 0:
                non_perm_obj_coords.append(np.array([i,j]))

    # ermittle Eckpunktkkordinaten über die längste Distanz
    # zum Flächenschwerpunkt

    #-> Flächenschwerpunkt (:= mittler Ortsvektor)
    focus_point = sum(non_perm_obj_coords)/len(non_perm_obj_coords)

    #-> finde Eckpunkte (da Rechteck => die vier tuple mit der größten Distanz zum Flächenschwerpunkt)
    DISTANCES   = [np.sqrt((focus_point[0]-point[0])**2 + (focus_point[1]-point[1])**2) for point in non_perm_obj_coords]

    # sortiere Distanzen nach absteigendem Prinzip
    DISTANCES   = sorted(DISTANCES)

    # extrahiere die Eckpunktkoordninaten aus der Liste aller Koordinaten
    max_dist    = DISTANCES[-1]


    corner_point_coords = []

    for point in non_perm_obj_coords:
        if np.sqrt((focus_point[0]-point[0])**2 + (focus_point[1]-point[1])**2) == max_dist:
            corner_point_coords.append(point)

    return corner_point_coords


def hydro_grad(perm):
    hydr_grad = np.ones((np.shape(perm)[0], np.shape(perm)[0]))

    slope = 0.0001
    for i in range(np.shape(perm)[0]):
        hydr_grad[i,:] = -slope*i

    return hydr_grad









class PermPatternGen:

    def __init__(self, src_path):
        self.src_path = src_path


    def load_im(self):

        # liste alle möglichen Eltern Permeabilitätspattern auf
        pattern_names      = os.listdir(self.src_path)

        # wähle zufällig eins aus diesen aus
        # rand_choise_mother = pattern_names[rand.randint(0,len(pattern_names)-1)]
        rand_choise_mother = pattern_names[rand.randint(0, len(pattern_names)-1)]

        # lade randomisiert eine parent-pattern-matrix
        return plt.imread(os.path.join(self.src_path,rand_choise_mother))
        

    def get_random_pattern_im(self):

        # parent_pattern = self.__grab_image()
        parent_pattern = self.load_im()

        

        def rand_rot(pattern_arr):
            #rotieren zufällig die Eltern-Pattern-Matrix bzgl der beiden Axen jeweils 90° oder 0°
            if rand.randint(0,1) == 1:
                arr_flip = np.flip(pattern_arr,rand.randint(0,1))
            else:
                arr_flip = pattern_arr
            arr_flip = np.flip(arr_flip,rand.randint(0,1))

            return arr_flip


        def rand_sampling():
            # plt.imshow(parent_pattern)
            # plt.show()

            # extremale sample Bereiche
            I_J__bound = [51,448]

            # 4 zufällige sample-points
            sample_point = (rand.randint(49, 449), rand.randint(49,449))
            # print(sample_point)



            child_pattern = np.ones((50,50))

            j_min = 0
            for j in range(sample_point[0]-24, sample_point[0]+26):
                k_min = 0
                for k in range(sample_point[1]-24, sample_point[1]+26):
                    # print(80*'~')
                    # print('j', j)
                    # print('k', k)
                    # print('j_min', j_min)
                    # print('k_min', k_min)
                    child_pattern[j_min, k_min] = parent_pattern[j, k]
                    k_min += 1
                j_min += 1

            

            # plt.imshow(child_pattern)
            # plt.show()

            return child_pattern

        def rand_smoother(arr):
            sigma_range = (1,4)

            return gaussian_filter(arr, sigma=rand.randint(sigma_range[0], sigma_range[1]))


        def normalizer(arr):
            '''
            Normalisiert das Array auf den Maximalwert = 1
            '''

            max_val = np.max(arr)

            return (1/max_val)*arr


   
        return rand_rot(normalizer(rand_smoother(rand_sampling())))




# perm_pattern_obj = PermPatternGen('./mother_perm_matrizes')

# perm_pattern_obj.get_random_pattern_im()

# for i in range(5):
#     im = perm_pattern_obj.get_random_pattern_im()
#     plt.imshow(im)
#     plt.show()



class Solver:
    def __init__(self, perm, non_perm_objects):
        '''
        Attribute sind alle relevanten hydrogeologischen Parameter. Da die Parameterfelder über Graustufenbilder
        abgebildet werden, sind nur Werte zwischen 0 und 1. Deshalb müssen einige Parameter rescaled werden 
        um stabile Lösungen zu gewährleisten. 
        '''
        self.perm         = perm              # Permeabilitätsfeld (Primärparameter in der Modellierung)
        #self.hydr_grad    = 0.1*hydr_grad     # upscaling, damit ein geringfügiger Effekt zu sehen ist. 
        # self.source       = source             # Quellterm (Inhomogenität in Form eines Massenflusses) >> wird nicht mitmodelliert <<
        self.diff         = 0.00006               # 0.1 ist maximaler Wert wenn, dt = [1, 1.5] und dx = 1
        self.non_perm_objects = non_perm_objects       # Nichtpermeables Objekt (Bauwerk/Spundwandwanne) 
        
        self.hydr_grad = hydro_grad(perm)
    

    def set_init_cond(self, init_cond_, dx, dt):
        '''
        Settet die aktuellen Anfangswerte.
        '''
        self.init_cond = init_cond_
        self.dx        = dx
        self.dt        = dt
        

    def solve(self):
        '''
        Solvermethode als Kern der Klasse. Löst die PDG über explizite Euler-Methode.
        '''
        C_old = self.init_cond
        D     = self.diff
        K     = self.perm
        H     = self.hydr_grad
        C_new = np.ones(np.shape(self.init_cond))
  
        dx = self.dx
        dt = self.dt

        
        for i in range(1,np.shape(self.init_cond)[0]-1):
            for j in range(1,np.shape(self.init_cond)[1]-1):
            
                # Zum modellieren von nichtpermeablen und nichtdiffusiven Objekten im 
                # Lösungsgebiet wird eine Nullgradientenbedingung auf die korrespondierenden 
                # Gitterzellen durch kopieren der Gitterwerte aus dem letzten Zeitschritt 
                # abgebildet.
                # Solche Objekte bzw. Gebiete werden durch die <non_perm_obj>-Matrix abgebildet
                # impermeable Zelle hat den Pixelwert 0 sonst 1 
                
                flag = 0

                if self.non_perm_objects != False:
                    for self.non_perm_object in self.non_perm_objects:
                        if self.non_perm_object[i,j] == 0:                   
                            #C_new[i, j] = C_old[i, j]
                            C_new[i, j] = 0
                            flag = 1


                if flag == 0:    
                  
                    H_old_i_j   = H[i,j]
                    H_old_ip1_j = H[i+1,j]
                    H_old_im1_j = H[i-1,j]
                    H_old_i_jp1 = H[i,j+1]
                    H_old_i_jm1 = H[i,j-1]

                    C_old_i_j   = C_old[i,j]
                    C_old_ip1_j = C_old[i+1,j]
                    C_old_im1_j = C_old[i-1,j]
                    C_old_i_jp1 = C_old[i,j+1]
                    C_old_i_jm1 = C_old[i,j-1]
                    
                    K_old_i_j   = K[i,j]
                    K_old_ip1_j = K[i+1,j]
                    K_old_im1_j = K[i-1,j]
                    K_old_i_jp1 = K[i,j+1]
                    K_old_i_jm1 = K[i,j-1]
                    
                                            
                    
                    term_a = D*(1/dx**2)*(C_old_ip1_j + C_old_im1_j + C_old_i_jp1 + C_old_i_jm1 - 4*C_old_i_j) 
                    term_b = C_old_i_j*(1/(4*dx**2))*(K[i+1,j] - K[i-1,j] + K[i,j+1] - K[i,j-1])*(H[i+1,j] - H[i-1,j] + H[i,j+1] - H[i,j-1])
                    term_c = C_old_i_j*(1/dx**2)*K[i,j]*(H_old_ip1_j + H_old_im1_j + H_old_i_jp1 + H_old_i_jm1 - 4*H_old_i_j)
                    term_d = K[i,j]*(1/(4*dx**2))*(C_old_ip1_j - C_old_im1_j + C_old_i_jp1 - C_old_i_jm1)*(H[i+1,j] - H[i-1,j] + H[i,j+1] - H[i,j-1])

                    #print(80*'~')
                    # if term_b > 0 or term_c > 0:non_perm_objects
                    #     print('wrogjerogjo')
                    # print(term_b)
                    # print(term_c)

                    #C_new[i, j] = dt*(term_a  + term_b  + term_c) + C_old_i_j
                    C_new[i, j] = dt*(term_a - term_b  - term_c - term_d) + C_old_i_j


                    #print((1/2*dx)*(H[i+1,j] - H[i-1,j] + H[i,j+1] - H[i,j-1]))
                    #C_new[i, j] = dt*(term_a) + C_old_i_j
                

                
        # Addiere Quelle -> Inhomogenitätsterm wird auf aktuelle Zeitlösung addiert. Quellmatrix ist 0 an jeder Stelle ungleich des Quellortes
        # und 1 am Ort (Koordinaten) der Quelle. Simulation von Stoffeintrag durch Massenfluss in das Simulationsgebiet.
        # C_new = C_new + self.source  


        # Randbedingung: Überschreibe Neue Randwerte mit den den alten Randwerten
        # -> Nullgradientenbedingung (wird aber nicht realisiert-> relativ starker Fluss aus den Rändern). 
        # Somit müssen auch keine Linksseitigen bzw. rechtsseitigen Differenzenquotienten
        # zum Approximieren an den Rändern genutzt werden. 

        C_new[:, 0] = C_old[:,0 ]
        C_new[:,-1] = C_old[:,-1]
        C_new[0, :] = C_old[0, :]
        C_new[-1,:] = C_old[-1,:]

        # C_new[:, 1] = C_old[:,1 ]
        # C_new[:,-2] = C_old[:,-2]
        # C_new[1, :] = C_old[1, :]
        # C_new[-2,:] = C_old[-2,:]

        return C_new



class Trainingsdata:

    def __init__(self, src_path):
        self.src_path = src_path


    def get_tdata(self):
        META_DATA = []
        INPUT     = []
        OUTPUT    = []

        with open(self.src_path, 'r') as fin:
            flag   =  'standby'
            flag_1 = 0
            for line in fin.readlines():
                if '<meta>' in line:
                    flag = 'get_meta_data'
                    continue

                if '</meta>' in line:
                    continue

                if '<data>' in line:
                    continue
                
                if '<case n>' in line:
                    flag = 'active'
                    sub_list = [[],[]]
                    continue

                if '</case n>' in line:
                    flag = 'standby'

                if '<permeability>' in line:
                    flag = 'get_input_vector'
                    continue

                if '</permeability>' in line:
                    flag = 'wait_for_output_vector'

                if '<sample values>' in line:
                    flag = 'get_output_vector'
                    continue

                if '</sample values>' in line:
                    flag = 'waiting_for_new_case'



                if flag == 'get_meta_data':
                    for form_char in ['\t', '\t\t', '\t\t\t', '\n']:
                        line = line.strip(form_char)

                    line = line.split(', ')
                    for item in line:
                        item = item.split('=')
                        META_DATA.append(int(item[1].replace('>','')))



                if flag == 'get_input_vector':
                    for form_char in ['\t', '\t\t', '\t\t\t', '\n']:
                        line = line.strip(form_char)

                    sub_list[0].append(float(line))

                if flag == 'get_output_vector':
                    for form_char in ['\t', '\t\t', '\t\t\t', '\n']:
                        line = line.strip(form_char)

                    sub_list[1].append(float(line))

                if flag == 'waiting_for_new_case':
                    INPUT.append(sub_list[0])
                    OUTPUT.append(sub_list[1])


        return torch.tensor(INPUT), torch.tensor(OUTPUT)

