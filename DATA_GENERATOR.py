import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy as cp
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('ggplot')
from matplotlib import rc
rc('font',**{'family':'serif','sans-serif':['Times']})
import matplotlib.patches as patches

import LIB as LIB

# obere Integrationsgrenze Zeit 
#T_ = 3600 
T_ = 260

# Anzahl der Simulationsfälle
N_cases = 20000

'''
Alle Parameter:
''''''''''''''''''

Zeitbereich            T    FIX
Zeitschritte           dt   FIX
Anfangswertfeld             FIX
Gebietsdimensionierung      FIX    (50x50)
Ortsdiskretisierung    dx   FIX
hydraulischer Gradient      FIX
Ort der Beprobung           FIX

Permeabilitätsfeld          VARIABEL
nichtdiffusive Bereiche     NICHT MODELLIERT  


'''

# Beprobungsort (Bleibt über alle Trainingsdaten konstant)
obs_well = (20, 20)




# initialisiere Anfangswert
init_cond    = 20*plt.imread('./Anfangsbedingung.png')


perm_parent_pattern_path = './mother_perm_matrizes'

source       = plt.imread('./source.png')

# non_perm_objects = []
# for fname in os.listdir('./non_diffusiv_objects'):
#     non_perm_objects.append(plt.imread('./non_diffusiv_objects/' + fname))

eps         = 0
dt          = 0.5
dx          = 0.1
T           = 3500

sol_counter = 0

n = 0
for case in range(N_cases):

    # lade randomisiertes Permeabilitätsfeld

    perm_obj = LIB.PermPatternGen(perm_parent_pattern_path)
    perm     = perm_obj.get_random_pattern_im()
    
    non_perm_objects = False


    # instantisiere den Solver mit allen vordefinierten Parameterfeldern
    solver_obj = LIB.Solver(perm, non_perm_objects)

    # initialisiere Solver mit dem Anfangswertfeld 
    solver_obj.set_init_cond(init_cond, dx, dt)

    # relevanter Zeitschritt für die Ausgabe
    t_increm = 10

    # Liste aller Zeitabhängigen Lösungen 
    RESULTS = []


    max_c = np.max(init_cond)
    # print(80*'~')


    oszi = False
    kum  = False  


    for t in range(T):
        #print(t)

        solution = solver_obj.solve()
        # überschreibe alte Anfangswerte mit zeitlich aktueller Lösung -> Neue Anfangswerte sind alte Lösung
        solver_obj.set_init_cond(solution, dx, dt)

        if t % t_increm == 0:
            if np.min(solution) < 0:
                oszi = True
            if np.max(solution) - eps > max_c:
                kum  = True

            if kum == True or oszi == True:

                # Stabilitätskriterium Maximumsnorm: Der maximale Gradient über Ort und Zeit muss in jedem Zeitschritt abnehmen
                # Falls Maximumsnorm von Lösung(t) größer als Maximumsnorm von Lösung(t-1) wird abgebrochen und ein neuer Fall 
                # initiert. Nummerischer Fehler wird hier mit eps = 0.0001 geschaetzt.
                # Weiteres Stabilitätskriterium Minimumsnorm: Bei ungünstigen Verhältnis der Diskretisierungen (x, t) zur Permäbilität 
                # und Diffusion würde es zu Ostzillation kommen (Neumann'sches Kriterium) -> deis würde negative Lösungen induzieren
                # Letzterer Fall sollte allerdings durch die Normierung der der Permeabilität auf den Maximalwert von 1 ausgeschlossen sein.
                # Das sich der Wert 1 in den vorhergehenden Tests als stabil gezeigt hat. 

                print('ACHTUNG: Keine stabile Lösung !!!')
                print('\tOszillation :', oszi)
                print('\tKummulation :', kum)
                plt.imshow(perm)
                plt.savefig('./SOLUTIONS/unstable/' + str(n) + '_.png', dpi=300)
                break
 


            else:
                max_c = np.max(solution)
                # print(max_c)
    
 
            result_tmin1 = solution


    im = plt.imshow(perm)

    # print('Stabile Lösung ereicht')
    # Schreibe sample der Konzentration c(x_sample,t_sample), sowie das korrespondierende Permeabilitätsfeld in die Trainingsdaten-Datei

    if kum == True or oszi == True:
        # with open('./SOLUTIONS/unstable/' + str(n) + '_.dat', 'w') as output:
        #     pass
        #plt.savefig('./SOLUTIONS/unstable/' + str(n) + '_.png', dpi=300)
        pass

    else:
        try:
            plt.savefig('./SOLUTIONS/stable/' + str(n) + '_.png', dpi=300)
        except:
            pass

        new_dir = './SOLUTIONS/output/sol_' + str(sol_counter)


        os.makedirs(new_dir, exist_ok=True)

        np.savetxt(new_dir + '/' + str(n) + '_perm_field.csv', perm, delimiter=',')
        np.savetxt(new_dir + '/' + str(n) + '_solution_field.csv', solution, delimiter=',')


        sol_counter += 1



    n += 1


        












