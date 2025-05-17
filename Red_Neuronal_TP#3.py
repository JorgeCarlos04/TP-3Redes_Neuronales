import sklearn.neural_network
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
from mlxtend.plotting import plot_decision_regions
#Definir datos aleatorios
np.random.seed(58)
n_muestras=1000
x=np.random.uniform(low=-10,high=10,size=( n_muestras, 2))

#Definicion del cuadrante 
y=((x[:,0] >0) & (x[:,1] >0)).astype(int) #Convierte el booleano en entero (0,1)

#Entrenamiento y prueba
x_Ent , x_Prueb , y_Ent , y_Prueb = sk.model_selection.train_test_split(x,y,test_size=0.2,random_state=9)

#Modelo utilizado 
modelo=sk.neural_network.MLPClassifier(
hidden_layer_sizes=(16,8),
activation='relu',
solver='adam',
max_iter=1000,
random_state=18
)

#Entrenaiento
modelo.fit(x_Ent,y_Ent)

#Prediccion 
y_Pred=modelo.predict(x_Prueb)

#Precision del modelo utilizado 
precision=sk.metrics.accuracy_score(y_Prueb,y_Pred)
print(f"La precicion mostradapor el modelo es de : {precision:.3f}")

#Graficar el modelo(Metodo:plot_decision_regions)
plt.figure(figsize=(9,9))
plot_decision_regions(x_Prueb , y_Prueb , clf=modelo, legend=2)
plt.title("Localizacion del punto segun el cuadrante ")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


