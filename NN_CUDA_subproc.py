import os.path
import torch
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt
import LIB
import time as time
import random as rand

name_handle = sys.argv[1] + '_' + sys.argv[2]

print(80*'=')
print('run NN calculation of case ', name_handle)
#print(sys.argv)




t_0 = time.time()

'''
CUDA...CoolerUltraDerbeAbgefahrener Scheiss

Man definiert mit <TENSOR_OBJEKT_X>.to(<graka_index>), dass das Tensorobjekt, anstatt über
die CPU gerechnet zu werden, über die Grafikkarte (<graka_index>) gerechnet wird. 
Dementsprechend müssen also alle in der Laufzeit entstehenden 
Tensorobjekte ebenfalls über die Grafikkarte gerechnet werden.
Die Grafikkarte wird als Objekt über <torch.device('cuda:0' if use_cuda else 'cpu')> definiert
wobei torch.cuda.is_available() testet ob eine CUDA-fähige Grafikkarte überhaupt existiert.
Die Grafikkarten werden als key-value-paare-strings definiert und mit n = 0 + i indiziert.

Der Geschwindigkeitszuwachs ist entgegen CPU-Nutzung (Trotz i-9!!!) erheblich...wohl aber auch der Stromverbrauch.
Die Grafikkarte hat sich nach 3000 bis 5000 Epochen auf etwa 75 °C erhitzt und gehalten. Die CPU Kerne hielten sich jeweils
bei etwa 50 °C. Jedoch immer nur bei ca 70 % Auslastung --> Overclocking nicht weiter sinnvoll.

'''



# frage ob CUDA möglich ist und gib es als Wahrheitsvariable aus
use_cuda = torch.cuda.is_available()

# wenn CUDA möglich ist, definiere, dass das Gerät <device> der ersten Grafikkarte zugeordnet wird
device   = torch.device('cuda:0' if use_cuda else 'cpu')

# falls über die CPU gerechnte wird, nutze n CPU-Kerne
if use_cuda == False:
    nutze CPU auf 15 Kernen
    device   = torch.device('cpu')
    torch.set_num_threads(15)


# lade die Trainingsdaten
X, Y = LIB.Trainingsdata('./' + name_handle + '.dat').get_tdata()


# splitte die Trainingsdaten zu je (ca.) 50 % in tatsächlichen Trainingsdaten und Validierungsdaten
X_training = X[:len(X)//2]
X_training = X_training.to(device) 

X_valid    = X[len(X)//2:]
X_valid    = X_valid.to(device) 

Y_training = Y[:len(Y)//2]
Y_training = Y_training.to(device) 

Y_valid    = Y[len(Y)//2:]
Y_valid    =  Y_valid.to(device) 





# Diverse Vorgaben zum Neuronalen Netz
Number_Input_Neurons  = len(X[0])                # Anzahl der Eingangsneuronen angepasst an die Größe der Trainingsdaten
Number_Output_Neurons = len(Y[0])                # Anzahl der Ausgabeneuronen
Learning_Rate         = 0.000001                # Lernrate (bei Verwendung des ADAM-Optimizers sollte diese locker unter 1e-5 sein)
Number_Epochs         = 1000                      # Anzahl der Durchläufe
Model_Parameter_File  = "model_status_cuda.pt"   # Dateiname für die Weights



# definiere  NN-Topologie  !!! Wichtig ist, dass das Objekt als CUDA-Tensorobjekt deklariert werden muss, wenn über die GPU gerechnet werden soll
model = nn.Sequential( 
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.Tanh(),
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.ReLU(),
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.Tanh(),
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.ReLU(),                                  
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.Tanh(),       


            nn.Linear(Number_Input_Neurons, 1)
).to(device)
 

# definiere Fehlerfunktion für Evaluierungszwecke
def err(true_Y, pred_Y, mode=None):

    err_sum = 0
    for true_val, model_val in zip(true_Y, pred_Y):
        if mode == 'abs':
            err_sum += abs(true_val - model_val)

        if mode == 'qrd':
            err_sum += (true_val - model_val)**2


    return err_sum
    



# # Lädt die gespeicherten Weights, wenn Datei vorhanden
# if os.path.isfile(Model_Parameter_File):
#     model.load_state_dict(torch.load(Model_Parameter_File))

# Loss Function vorgeben (Mean Squared Error Loss)
criterion = torch.nn.MSELoss(reduction='sum')

# Optimizer vorgeben (SGD Stochastic Gradient Descent)  # hat sich für dieses Model als signifikant 
# schlecht gegenüber dem ADAM-Optimizer herausgestellt
#optimizer = torch.optim.SGD(model.parameters(), lr = Learning_Rate, momentum=0.9)

# ADAM Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)

# Das ist jetzt die eigentliche Ausführung des Prozesses

# Liste zum sammeln aller Modellfehler hinsichtlich sowohl der Prädiktionsfehler als auch der Fehler hinsichtlich
# der Validierungsadten, für anschließende Evaluierungszwecke
ERRs     = []
ERR_rand = []



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Optimierungsprozess des NN

# Iteriere über die vorgegebene Anzahl an Epochen
for epoch in range(Number_Epochs):


    # Vorwärts Propagierung
    Y_predicted = model(X_training).to(device) 

    # Berechne Fehler und gebe ihn aus
    loss = criterion(Y_predicted, Y_training)
    #print('epoch: ', epoch, ' loss: ', loss.item())

    # wende Gradientenabstiegsmethode auf den Optimizer an
    optimizer.zero_grad()

    # Führe Modellfehler rückwertig auf das Modell zurück
    loss.backward()

    # Update the parameters
    optimizer.step()


    ###### Evolution

    Y_valid_model = model(X_valid)

    err_pred_vs_training = float(err(Y_training, Y_predicted, mode='abs')) 
    err_pred_vs_valid    = float(err(Y_valid, Y_valid_model, mode='abs'))

    MAE_training         = err_pred_vs_training/len(Y_training)
    MAE_valid            = err_pred_vs_valid/len(Y_valid)

    ERRs.append([MAE_training, MAE_valid])

    rand_idx = rand.randint(0,len(Y_training)-1)

    rand_true_y = float(Y_valid[rand_idx])
    rand_pred_y = float(Y_valid_model[rand_idx])

    ERR_rand.append(abs(rand_true_y-rand_pred_y))

    # print(80*'~')
    # print(epoch)
    # print('Training_err_abs  : ', err_pred_vs_training)
    # print('Validation_err_abs: ', err_pred_vs_valid)
    #print('-------------------')
    #print('Training_err_qrd  : ', err_pred_vs_training_qrd)
    #print('Validation_err_qrd: ', err_pred_vs_valid_qrd)
    # print('-------------------')
    # print('random_valid:')
    # print('True y            :', rand_true_y)
    # print('pred y            :', rand_pred_y) 
    # print('abs_err           :', abs(rand_true_y-rand_pred_y))



# Plotte Ergebnisse zur Validierung des Modells

epoch_n = 1
flag    = 0
epochs = []
for sublist in ERRs:

    if flag == 0:

        # plt.scatter(epoch_n, sublist[0], color='green', label='Training')
        # plt.scatter(epoch_n, sublist[1], color='blue', label='Validation')
        #plt.scatter(epoch_n, ERR_rand[epoch_n-1], color='red', label='rand_validation')
        epochs.append(epoch_n)
        flag = 1
        continue

    else:
        # plt.scatter(epoch_n, sublist[0], color='green')
        # plt.scatter(epoch_n, sublist[1], color='blue')
        epochs.append(epoch_n)
    


    epoch_n += 1

# plt.plot(epochs, ERR_rand, color='red', alpha=0.5)
# print(80*'=' + '\n')
# print(str(time.time()-t_0))


# plt.legend()
# plt.show()

#torch.save(model.state_dict(), Model_Parameter_File)




np.savetxt('./Konvergenzstudie/' + name_handle + '.csv', np.asarray(ERRs), delimiter=',')

with open('./Konvergenzstudie/' + name_handle + '_rand_val.csv', 'w') as fout:

    for x,y in zip(epochs, ERR_rand):
        fout.write(str(x) + ';' + str(y) + '\n')


print('NN calculation for case ', name_handle, ' done in ', str(round(time.time()-t_0)), 'seconds')

