import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd 
from pywt import wavedec
import scipy.signal as sg
import seaborn as sns

def get_data(subject="s10",activity="run"):
    cwd=Path.cwd()

    DATA_DIR=cwd/"data"

    df=pd.read_csv(DATA_DIR/f"{subject}_{activity}.csv")
    
    return df

def plot_signal(x,y):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    ax.plot(x,y)  # Plot some data on the axes.

    
    plt.grid(visible=True, which='major', color='k', linestyle='-',alpha=0.3)
    plt.grid(visible=True, which='minor', color='k', linestyle='-',alpha=0.1)
    plt.minorticks_on()
    
def display_signal(t_inicial=300,signal="ecg"):

    df=get_data(subject="s10",activity="run")
    N=2000
    Ts=0.002
    n_ini=int(t_inicial/Ts)
    t=np.arange(0,N+1)*Ts+t_inicial
    plot_signal(t,df.loc[n_ini:n_ini+N,signal])
    plt.title(signal)
    plt.xlim([0+t_inicial,N*Ts+t_inicial])
    plt.xlabel("time[sec]")
    
def plot_coeffs(coeffs):
    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(5, 1, figsize=(8, 12))  # 5 rows of subplots

    # Plot data on the first subplot
    axs[0].plot(coeffs[0], color='blue')
    axs[0].set_title('C_a')

    # Plot data on the second subplot
    axs[1].plot(coeffs[1], color='red')
    axs[1].set_title('Cd4')

    # Plot data on the third subplot
    axs[2].plot(coeffs[2], color='green')
    axs[2].set_title('Cd3')

    # Plot data on the fourth subplot
    axs[3].plot(coeffs[3], color='orange')
    axs[3].set_title('Cd2')

    # Plot data on the fifth subplot
    axs[4].plot(coeffs[4], color='purple')
    axs[4].set_title('Cd1')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def get_wavelet_correlation_modes(x,plot):
    coeffs = wavedec(x, 'db4', level=4)

    coeffs=np.array([sg.resample(coeff,len(coeffs[-1])) for coeff in coeffs])
    corr=coeffs@coeffs.T
    if plot:
        # Create a figure and a grid of subplots
        fig, axs = plt.subplots(6, 1, figsize=(8, 12))  # 5 rows of subplots

        # Plot data on the first subplot
        axs[0].plot(coeffs[0], color='blue')
        axs[0].set_title('C_a')

        # Plot data on the second subplot
        axs[1].plot(coeffs[1], color='red')
        axs[1].set_title('Cd4')

        # Plot data on the third subplot
        axs[2].plot(coeffs[2], color='green')
        axs[2].set_title('Cd3')

        # Plot data on the fourth subplot
        axs[3].plot(coeffs[3], color='orange')
        axs[3].set_title('Cd2')

        # Plot data on the fifth subplot
        axs[4].plot(coeffs[4], color='purple')
        axs[4].set_title('Cd1')

        axs[5] = sns.heatmap(corr, linewidth=0.5,annot=True)
        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plot
        plt.show()
    return corr
    
def classifier_only_ecg(df_wcm):

    #df_wcm=pd.read_csv("wcm.csv")
    df_y,df_x=df_wcm["out"],df_wcm.iloc[:,2:27]
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
     
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
    
    scaler = StandardScaler()
    df_x_t=scaler.fit_transform(df_x)

    df_x_t_pca = PCA(n_components=25).fit_transform(df_x_t)
    
    y= df_y.values.ravel()
    X=df_x_t_pca
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, y)
    y_pred=clf.predict(X)
   
    cm=confusion_matrix(y, y_pred)

    

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Sentado","Caminando","Corriendo"],
    )
    disp.plot(cmap=plt.cm.Blues)
    
    from sklearn.metrics import classification_report
    print(classification_report(
        y, y_pred, 
        target_names=["Sentado","Caminando","Corriendo"]
    ))