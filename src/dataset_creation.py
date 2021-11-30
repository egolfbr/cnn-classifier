import keras 
import numpy as np
from keras import layers
import chirp
import matplotlib.pyplot as plt




def create_dataset():
    T = 5
    x = np.linspace(0,1,T*1000)
    sin_x = np.linspace(0,1,T*1000)
    for i in range(1,10000):
        init_freq = i
        fin_freq = i*10
        carrier = i*10
        mod = i
        y = np.zeros(5000)
        if i > 100 and i < 500:
            sin_x = np.linspace(0,0.1,T*1000)
        if i > 500 and i < 1000:
            sin_x = np.linspace(0,0.01,T*1000)
        if i > 1000 and i < 10000:
            sin_x = np.linspace(0,0.001,T*1000)
        
        y, ft_lin = chirp.linearChirp(init_freq,fin_freq,5,x)
        y, ft_sin = chirp.sinusoidalMod(carrier,mod,sin_x)
        y, ft_geo = chirp.geometricChirp(init_freq,fin_freq,2,x)
        fig = plt.plot(x,ft_lin)
        ax=plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig('./ece529_repository/project/imgs/lin_'+str(i)+'.png', bbox_inches='tight',pad_inches=0)
        plt.close() 
        fig = plt.plot(x,ft_geo)
        ax=plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig('./ece529_repository/project/imgs/geo_'+str(i)+'.png', bbox_inches='tight',pad_inches=0)
        plt.close() 
        fig = plt.plot(x,ft_sin)
        ax=plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig('./ece529_repository/project/imgs/sin_'+str(i)+'.png', bbox_inches='tight',pad_inches=0)
        plt.close() 
    print("Dataset Complete")

def main():
    create_dataset()
    print("Complete")
    
main()
