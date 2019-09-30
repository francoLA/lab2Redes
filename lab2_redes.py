import numpy as np
import matplotlib.pyplot as plot
from scipy.io import wavfile
from scipy.fftpack import fft as transformadaFourier
from scipy.fftpack import fftfreq as funcionFrecuencia
from scipy.fftpack import ifft as transformadaFourierInversa
from scipy import signal as sg
from math import log

################ LECTURA DEL AUDIO HANDEL.WAV  ##################################################################################

# Se reconoce el archivo de audio handel y se guarda, Donde datos es un arreglo que contiene las amplitudes
#freqMuestreo: Corresponde a la cantidad de mediciones por segundo que se hicieron en la lectura del archivo .wav
#datos: Corresponde a los datos (amplitudes) registradas en el audio leido.

(freqMuestreo, datos) = wavfile.read('handel.wav')
vectorTiempoAudio= np.linspace(0, len(datos)/freqMuestreo, num=len(datos))
frecuencias = funcionFrecuencia(len(datos), 1.0 /freqMuestreo)

#################################    ESPECTOGRAMA DEL AUDIO ORIGINAL    ##########################################################
def transformadaFourierYgrafico(freqMuestreo,datos):

    #Se genera el espectograma del audio
    f,t,Sxx = sg.spectrogram(datos,freqMuestreo)
    #Grafico del espectograma
    plot.figure('Dominio de la frecuencia en el tiempo')
    plot.pcolormesh(t,f,np.log10(Sxx))
    plot.xlabel('Tiempo(s)')
    plot.ylabel('Frecuencia(Hz)')
    plot.colorbar()
    plot.show()

##################################   FILTRO FIR    ###########################################################################
def filtroFir(datos, freqMuestreo, frecuencias):

    #Se genera el filtro pasa bajos
    filtroAlto = sg.firwin(50, 1500, fs = freqMuestreo)

    #Se aplica el filtro al audio
    filtrado = sg.lfilter(filtroAlto, [1, 0], datos)

    #Se calcula la transformada de Fourier
    filtroTransformada = transformadaFourier(filtrado)

    a = filtrado

    #Grafico en el dominio de la frecuencia
    plot.figure('Dominio de la frecuencia pasa bajos')
    plot.xlabel('Frecuencia(Hz)')
    plot.ylabel('F(w)')
    plot.plot(frecuencias, abs(filtroTransformada))
    plot.show()

    señal = np.asarray(transformadaFourierInversa(filtroTransformada), dtype=np.int16)
    wavfile.write('filtroPasaAltos.wav', freqMuestreo, señal)

    return a

def filtroTiempo(filtrado, freqMuestreo):

    #Se genera el espectograma despues del filtrado
    f, t, Sxx = sg.spectrogram(filtrado, freqMuestreo)

    #Grafico en el dominio del tiempo
    plot.figure('Dominio de la frecuencia pasa bajos')
    plot.pcolormesh(t, f, np.log10(Sxx))
    plot.xlabel('Tiempo(s)')
    plot.ylabel('Frecuencia(Hz)')
    plot.colorbar()
    plot.show()

############################ BLOQUE PRINCIPAL O MAIN ################################

transformadaFourierYgrafico(freqMuestreo,datos)
inv = filtroFir(datos, freqMuestreo, frecuencias)
filtroTiempo(inv, freqMuestreo)