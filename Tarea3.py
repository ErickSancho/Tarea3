# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:03:18 2020

@author: Alonso Sancho
"""

import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot  as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D    #Para lo graficos 3d

def separarletra(val):
    entero=0
    for i in range(1,len(val)):
        entero=entero*10
        entero+=int(val[i])        
    return entero

def separar(lista):
    vector=[]
    for etr in lista:
        if len(etr)>0:
            vector.append(separarletra(etr))
        else:
            continue
    return vector


datos=[]
valoresX=[]
valoresY=[]
with open('xy.csv', newline='') as File:  #Abrimos el documento
    lineas = csv.reader(File)               #Extraemos cada linea en una lista
    for row in lineas:
        datos.append(row)


datos_mod=np.array(datos)

valoresX=separar(datos_mod[:,0])
valoresY=separar(datos_mod[0,:])

datos_mod=np.delete(datos_mod, [0][0], axis=1)
datos_mod=np.delete(datos_mod, [0][0], axis=0)

datos_mod=np.array(datos_mod, dtype=np.float64)

sumaY=np.sum(datos_mod, axis=0)
sumaX=np.sum(datos_mod, axis=1)

print("La suma para el eje x para la funcion de densidad de X es: ", np.sum(sumaX))
print("La suma para el eje y para la funcion de densidad de Y es: ", np.sum(sumaY))

plt.plot(valoresY, sumaY)
plt.title("Funcion de densidad marginal para Y")
plt.savefig("Graficas/Grafica_para_Y.png")
plt.cla()

plt.plot(valoresX, sumaX)
plt.title("Funcion de densidad marginal para X")
plt.savefig("Graficas/Grafica_para_X.png")
plt.cla()

########################
#De las graficas calculadas se puede suponer que las distribuciones de las funciones de densidad marginal son de 
#tipo gaussiana, por lo tanto, defino una funcion que calcula los valores de un distribucion normal, para poder 
#compararla con las funciones de densidad marginal calculadas.

def gaussiana(x, a, b):
    return (1/(np.sqrt(2*(np.pi*(b*b)))))*np.exp(-(((x-a)*(x-a))/(2*(b*b))))


#Parametros para X
paramX,_=curve_fit(gaussiana,valoresX,sumaX)
print("Los parametros del ajuste de distribucion normal para x son: ",paramX)

#Parametros para Y
paramY,_=curve_fit(gaussiana,valoresY,sumaY)
print("Los parametros del ajuste de distribucion normal para y son: ", paramY)

#Genero los valores para los plots para cada funcion de densidad marginal

Xys=gaussiana(valoresX,paramX[0],paramX[1])

Yys=gaussiana(valoresY,paramY[0],paramY[1])

#Genero los plots
plt.plot(valoresX, sumaX)
plt.plot(valoresX,Xys)
plt.title("Curva de ajuste gaussiana para X")
plt.savefig("Graficas/Grafica_para_X_Curva_de_ajuste_normal.png")
plt.cla()


plt.plot(valoresY, sumaY)
plt.plot(valoresY,Yys)
plt.title("Curva de ajuste gaussiana para X")
plt.savefig("Graficas/Grafica_para_Y_Curva_de_ajuste_normal.png")
plt.cla()

######################## Prueba para Rayleigh


def rayleigh(x, s):
    return (x/(2*pow(s,2)))*np.exp((-(pow(x,2)))/(2*pow(s,2)))

#Parametros para X
paramX_rayleigh,_=curve_fit(rayleigh,valoresX,sumaX)
print("Los parametros del ajuste de distribucion rayleigh para x son: ", paramX_rayleigh)

#Parametros para Y
paramY_rayleigh,_=curve_fit(rayleigh,valoresY,sumaY)
print("Los parametros del ajuste de distribucion rayleigh para y son: ", paramY_rayleigh)

#Genero los valores para los plots para cada funcion de densidad marginal

Xys_rayleigh=[]
Yys_rayleigh=[]

for i in range(0,len(valoresX)):
    Xys_rayleigh.append(rayleigh(valoresX[i], paramX_rayleigh[0]))    

for i in range(0,len(valoresY)):
    Yys_rayleigh.append(rayleigh(valoresY[i], paramY_rayleigh[0]))


#Genero los plots
plt.plot(valoresX, sumaX)
plt.plot(valoresX,Xys_rayleigh)
plt.title("Curva de ajuste rayleigh para X")
plt.savefig("Graficas/Grafica_para_X_Curva_de_ajuste_rayleigh.png")
plt.cla()


plt.plot(valoresY, sumaY)
plt.plot(valoresY,Yys_rayleigh)
plt.title("Curva de ajuste rayleigh para Y")
plt.savefig("Graficas/Grafica_para_Y_Curva_de_ajuste_rayleigh.png")
plt.cla()

############################# Prueba para exponencial


def exponencial(x,a):
    return a*np.exp(-a*x)

#Parametros para X
paramX_exp,_=curve_fit(exponencial,valoresX,sumaX)
print("Los parametros del ajuste de distribucion exponancial para x son: ", paramX_exp)

#Parametros para Y
paramY_exp,_=curve_fit(exponencial,valoresY,sumaY)
print("Los parametros del ajuste de distribucion exponancial para y son: ", paramY_exp)

#Genero los valores para los plots para cada funcion de densidad marginal

Xys_exp=[]
Yys_exp=[]

for i in range(0,len(valoresX)):
    Xys_exp.append(exponencial(valoresX[i], paramX_exp[0]))    

for i in range(0,len(valoresY)):
    Yys_exp.append(exponencial(valoresY[i], paramY_exp[0]))




#Genero los plots
plt.plot(valoresX, sumaX)
plt.plot(valoresX,Xys_exp)
plt.title("Curva de ajuste exponencial para X")
plt.savefig("Graficas/Grafica_para_X_Curva_de_ajuste_exp.png")
plt.cla()


plt.plot(valoresY, sumaY)
plt.plot(valoresY,Yys_exp)
plt.title("Curva de ajuste exponancial para Y")
plt.savefig("Graficas/Grafica_para_Y_Curva_de_ajuste_exp.png")
plt.cla()


##################################

def uniforme(x,a):
    return 1/a

#Parametros para X
paramX_uni,_=curve_fit(uniforme,valoresX,sumaX)
print("Los parametros del ajuste de distribucion uniforme para x son: ", paramX_uni)

#Parametros para Y
paramY_uni,_=curve_fit(uniforme,valoresY,sumaY)
print("Los parametros del ajuste de distribucion uniforme para y son: ", paramY_uni)

#Genero los valores para los plots para cada funcion de densidad marginal

Xys_uni=[]
Yys_uni=[]

for i in range(0,len(valoresX)):
    Xys_uni.append(uniforme(valoresX[i], paramX_uni[0]))    

for i in range(0,len(valoresY)):
    Yys_uni.append(uniforme(valoresY[i], paramY_uni[0]))




#Genero los plots
plt.plot(valoresX, sumaX)
plt.plot(valoresX,Xys_uni)
plt.title("Curva de ajuste uniforme para X")
plt.savefig("Graficas/Grafica_para_X_Curva_de_ajuste_uni.png")
plt.cla()


plt.plot(valoresY, sumaY)
plt.plot(valoresY,Yys_uni)
plt.title("Curva de ajuste uniforme para Y")
plt.savefig("Graficas/Grafica_para_Y_Curva_de_ajuste_uni.png")
plt.cla()

#######################################
#Calculo de la Covarianza, Correlaccion y coeficiente de Correlacion a partir de los datos
######################################

print('')
#Calculo de la media a partir de los datos:
media_x=0
media_y=0

for i in range(0,len(valoresX)):
    media_x+=valoresX[i]*sumaX[i]

for i in range(0,len(valoresY)):
    media_y+=valoresY[i]*sumaY[i]

print("La media de X es: ", media_x)
print("La media de Y es: ", media_y)

#Calculo de la variaza
varianza_x=0
varianza_y=0

for i in range(0,len(valoresX)):
    varianza_x=((valoresX[i])**2)*sumaX[i] - media_x

for i in range(0,len(valoresY)):
    varianza_y=((valoresY[i])**2)*sumaY[i] - media_y

print("La varianza de x es: ", varianza_x)
print("La varianza de y es: ", varianza_y)

#Calculo de la desviacion
desviacion_x=np.sqrt(varianza_x)
desviacion_y=np.sqrt(varianza_y)

print("La desviacion de la variable X es: ", desviacion_x)
print("La desviacion de la variable y es: ", desviacion_y)

#Calculo de la Correlacion

correlacion=0
for i in range(0,len(valoresX)):
    for j in range(0,len(valoresY)):
        correlacion+=valoresX[i]*valoresY[j]*datos_mod[i][j]
print("La correlacion de las variables X y Y es: ", correlacion)


#Calculo de la Coarianza
covarianza=correlacion-(media_x*media_y)
print("La covarianza para las variables X y Y es: ",covarianza)

#Calculo del coeficiente de pearson
pearson=(covarianza)/(desviacion_x*desviacion_y)
print("El coeficiente de correlacion (Pearson) es: ", pearson)


#graficas en 3d


datos_3col=[]
with open('xyp.csv', newline='') as File:  #Abrimos el documento
    lineas = csv.reader(File)               #Extraemos cada linea en una lista
    next(lineas)
    for row in lineas:
        datos_3col.append(row)
        
x=[int(datos_3col[x][0]) for x in range(0,len(datos_3col)) ]
y=[int(datos_3col[x][1]) for x in range(0,len(datos_3col)) ]
z=[float(datos_3col[x][2]) for x in range(0,len(datos_3col))]


# Ahora calculamos la gráfica en *3d* de la función de densidad conjunta a partir de los datos del archivo `xyp.csv`

fig = plt.figure() #Creo la figura
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0)


ax.set_xlim(5,15)
ax.set_ylim(5,25)
ax.set_zlim(0,0.016)

#Nombro los ejes y el grafico
plt.title("Funcion de densidad conjunta a partir de los datos")
plt.xlabel("Valores de X")
plt.ylabel("Valores de Y")
fig.tight_layout()

plt.savefig("Graficas/Superficie_densidad_conjunta.png")

plt.cla()

def gaussianaR2(x,y,ax,ay,bx,by):
    #tpx=(-(x-ax)**2)/(2*bx)
    #tpy=(-(y-ay)**2)/(2*by)
    return(1/(2*np.pi*bx*by))*np.exp((-(x-ax)**2)/(2*bx)+(-((y-ay)**2)/(2*by)))

#Creo el vector que contiene los valores de la funcion de densidad conjunta del modelo
z2=[]
for i in range(0,len(x)):
    z2.append(gaussianaR2(x[i], y[i], paramX[0], paramY[0], paramX[1], paramY[1]))

#Creo la figura que contiene el grafico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(x, y, z2, cmap=cm.jet, linewidth=0)


ax.set_xlim(5,15)
ax.set_ylim(5,25)
ax.set_zlim(0,0.016)

#Defino los limites de los ejes
ax.set_xlim(5,15.5)
ax.set_ylim(5,25.5)
ax.set_zlim(0,np.max(z2)+0.0005)

#Nombro los ejes
plt.xlabel("Valores de X")
plt.ylabel("Valores de Y")
plt.title("Funcion de densidad conjunta a partir del modelo calculado")
fig.tight_layout()

plt.savefig("Graficas/Superficie_densidad_conjunta_del_modelo.png")
plt.cla()

