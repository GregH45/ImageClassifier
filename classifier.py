import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from matplotlib.ticker import FuncFormatter
from time import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Donnees d'entree
trn_img = np.load('data/trn_img.npy')
trn_lbl = np.load('data/trn_lbl.npy')
dev_img = np.load('data/dev_img.npy')
dev_lbl = np.load('data/dev_lbl.npy')
tst_img = np.load('data/tst_img.npy')
class_number = 10



'''            QUESTION 1    
    Classifieur Bayesien Gaussien '''
 
      
''' Creation de la base d'apprentissage '''

def learning(X, Y):
    representants = []
    covariances_log = []
    covariances_inv = []
    
    # Pour chaque classes possible  
    for i in range(class_number):
        
        # On calcul ses representants -> moyenne des vecteurs d'apprentissage
        representants.append(X[Y==i].mean(axis=0))
        # On calcul sa matrice de covariance
        covariance = np.cov(X[Y==i],rowvar=0)
        
        # On prepare les futurs calculs de probabilites
        covariances_log.append(np.linalg.slogdet(covariance)[1])
        covariances_inv.append(np.linalg.inv(covariance))
    
    return representants, covariances_log, covariances_inv


''' Implementation du classifieur Bayesien Gaussien '''

def gaussian_bayesian_classifier(images, rep, log, inv):
    results = []
    
    # Pour chaque images
    for img in images:
        probabilites = []
        
        # Pour chaque classe possible
        for i in range(class_number):
            # On observe la ressemblance de l'image par rapport a celles de la classe  
            diff = (img - rep[i])   
            # On calcul sa probabilite d'appartenir a la classe
            probabilites.append(-log[i] -  np.dot(np.dot(np.transpose(diff), inv[i]), diff))
        
        # On choisit la plus grande probabilite trouvee pour classer l'image
        results.append(np.argmax(np.array(probabilites)))   
        
    return results
    

''' Execution de l'algorithme ''' 

def launch_bayesian_classifier(trn_img, trn_lbl, img_to_classify):
    
    # Lancement de l'apprentissage
    print("Apprentissage..")
    rep, log, inv = learning(trn_img, trn_lbl)
    
    # Lancement du classifieur
    print("Traitement..")
    start_time = time()
    results = gaussian_bayesian_classifier(img_to_classify, rep, log, inv)
    end_time = time()
    
    # Calcul du taux d'erreur
    errors_number = 0
    for index, result in enumerate(results) : 
        if result != dev_lbl[index] : errors_number += 1
    fail_rate = errors_number/img_to_classify.shape[0]
    print("Taux D'erreur:", fail_rate*100, "%")
    
    # Calcul du temps d'execution 
    execution_time = end_time-start_time
    print("Temps d'Execution:", execution_time, "s \n")
    
    return fail_rate, execution_time

def question1():
    launch_bayesian_classifier(trn_img, trn_lbl, dev_img)
    


'''            QUESTION 2    
        Reduction des vecteurs       '''
        
def question2():

    fail_rates = []
    executions_time = []
    tested_sizes = range(100, dev_img.shape[1], 100)
    
    for index, size in enumerate([25,50,75] + list(tested_sizes)):
        
        print("TEST", index, " - Classifieur Bayésien avec ACP de Taille ", size)
        
        # Definition de la taille de vecteur voulue
        pca_instance = PCA(size)
        
        # Transformation des vecteurs d'images 
        pca_instance.fit(dev_img)
        pca_instance.fit(trn_img)
        tnr_img_PCA = pca_instance.transform(trn_img)
        dev_img_PCA = pca_instance.transform(dev_img)
        
        # Lancement du classifieur gaussien sur les images reduites
        fail_rate, execution_time = launch_bayesian_classifier(tnr_img_PCA, trn_lbl, dev_img_PCA)
        
        # Enregistrement des donnes d'execution
        fail_rates.append(fail_rate)
        executions_time.append(execution_time)
    
    # Creation du graphique
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.set_xlabel("ACP Size")
    ax1.set_ylabel("Failure Rate", color='m')
    ax2.set_ylabel("Execution Time - seconds", color='c')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    
    ax1.plot([25,50,75] + list(tested_sizes),fail_rates,'m-x' )
    ax2.plot([25,50,75] + list(tested_sizes),executions_time,'c-*' )
    
    plt.show()



'''                    QUESTION 3    
        Comparaisons de differents classifieurs    '''

def question3():
    
    bayesian_fail_rates = []
    bayesian_executions_time = []

    svc_fail_rates = []
    svc_executions_time = []
    
    neighbors_fail_rates = []
    neighbors_executions_time = []
    
    tested_sizes = [25,50,75,100]
        
    for size in tested_sizes :
        
        print("Classifieurs Support Vector Machines et KNeighbors de parametre 3 avec ACP de Taille ", size)
        # Definition de la taille de vecteur voulue
        pca_instance = PCA(size)
            
        # Transformation des vecteurs d'images 
        pca_instance.fit(dev_img)
        pca_instance.fit(trn_img)
        tnr_img_PCA = pca_instance.transform(trn_img)
        dev_img_PCA = pca_instance.transform(dev_img)
    
        # Normalisation du jeu de donnees pour le classifieur SVM (moyenne)
        tnr_img_PCA_SCALE = scale(tnr_img_PCA)
        dev_img_PCA_SCALE = scale(dev_img_PCA)
    
        print("Apprentissages..")
        svc_instance = SVC()
        svc_instance.fit(tnr_img_PCA_SCALE, trn_lbl)
        
        neighbors_instance = KNeighborsClassifier(n_neighbors=3)
        neighbors_instance.fit(tnr_img_PCA, trn_lbl)
    
        print("Traitements..")
        start_time_svc = time() 
        svc_results = svc_instance.predict(dev_img_PCA_SCALE)
        end_time_svc = time() 
        
        start_time_neighbors = time() 
        neighbors_results = neighbors_instance.predict(dev_img_PCA)
        end_time_neighbors = time() 
        
        # Calcul du taux d'erreur
        errors_number = 0
        for index, result in enumerate(svc_results) : 
            if result != dev_lbl[index] : errors_number += 1
        svc_fail_rate = errors_number/dev_img.shape[0]
        print("Taux D'erreur SVC:", svc_fail_rate*100, "%")
        
        errors_number = 0
        for index, result in enumerate(neighbors_results) : 
            if result != dev_lbl[index] : errors_number += 1
        neighbors_fail_rate = errors_number/dev_img.shape[0]
        print("Taux D'erreur Neighbors:", neighbors_fail_rate*100, "%")
    
        # Calcul du temps d'execution 
        execution_time_svc = end_time_svc-start_time_svc
        print("Temps d'Execution SVC:", execution_time_svc, "s")
        
        execution_time_neighbors = end_time_neighbors-start_time_neighbors
        print("Temps d'Execution Neighbors:", execution_time_neighbors, "s \n")
 
        # Lancement du classifieur gaussien sur les images reduites
        print("Classifieur Bayésien avec ACP de Taille ", size)
        bayesian_fail_rate, bayesian_execution_time = launch_bayesian_classifier(tnr_img_PCA, trn_lbl, dev_img_PCA)
                
        # Enregistrement des donnes d'execution
        svc_fail_rates.append(svc_fail_rate)
        svc_executions_time.append(execution_time_svc)
        
        neighbors_fail_rates.append(neighbors_fail_rate)
        neighbors_executions_time.append(execution_time_neighbors)

        bayesian_fail_rates.append(bayesian_fail_rate)
        bayesian_executions_time.append(bayesian_execution_time)
    
    # Creation du graphique
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
        
    # Editions des labels
    ax1.set_xlabel("ACP Size")
    ax1.set_ylabel("Failure Rate")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    ax2.set_ylabel("Execution Time - seconds")
    
    # Edition de la legende 
    svc_lines = mlines.Line2D([], [], color="green", marker='x', label='SVC results')
    neighbors_lines = mlines.Line2D([], [], color="red", marker='x', label='Neighbors results')
    bayesian_lines = mlines.Line2D([], [], color="blue", marker='x', label='Bayesian results')
    exec_time_lines = mlines.Line2D([], [], color="black", linestyle="--", label='Execution Time')
    plt.legend(handles=[svc_lines, neighbors_lines, bayesian_lines, exec_time_lines])
    
    # Remplissage avec les donnees de resultat
    ax1.plot(tested_sizes,svc_fail_rates,'g-x')
    ax2.plot(tested_sizes,svc_executions_time,'g-x', linestyle=':')
    
    ax1.plot(tested_sizes,neighbors_fail_rates,'r-x')
    ax2.plot(tested_sizes,neighbors_executions_time,'r-x', linestyle=':')
    
    ax1.plot(tested_sizes,bayesian_fail_rates,'b-x')
    ax2.plot(tested_sizes,bayesian_executions_time,'b-x', linestyle=':')
        
    # Affichage du graphique
    plt.show()  
    
    
       
'''                            CHOIX DU MEILLEUR    
                                  CLASSIFIEUR  
            Matrice de confusion et classement des images de test            '''   
    
def best_classifier():
    
    print("Classifier Support Vector Machines with PCA Size 50 is the best! ")
    pca_instance = PCA(50)
        
    # Transformation des vecteurs d'images 
    pca_instance.fit(dev_img)
    pca_instance.fit(trn_img)
    pca_instance.fit(tst_img)
    tnr_img_PCA = pca_instance.transform(trn_img)
    dev_img_PCA = pca_instance.transform(dev_img)
    tst_img_PCA = pca_instance.transform(tst_img)

    # Normalisation du jeu de donnees pour le classifieur SVM (moyenne)
    tnr_img_PCA_SCALE = scale(tnr_img_PCA)
    dev_img_PCA_SCALE = scale(dev_img_PCA)
    tst_img_PCA_SCALE = scale(tst_img_PCA)
    
    svc_instance = SVC()
    svc_instance.fit(tnr_img_PCA_SCALE, trn_lbl)
    svc_results = svc_instance.predict(dev_img_PCA_SCALE)
    
    # Matrice de confusion 
    confusion_mat = confusion_matrix(dev_lbl, svc_results)
    df_cm = pd.DataFrame(confusion_mat, index = [i for i in range(10)], columns = [i for i in range(10)])
    plt.figure(figsize = (10,7))
    plt.title("Confusion Matrix")
    sn.heatmap(df_cm, annot=True)    
    
    #Classification images de tests
    np.save('test.npy', svc_instance.predict(tst_img_PCA_SCALE))
    
    # Verifications 
    tst_lbl = np.load('test.npy')
    
    fig, ax1 = plt.subplots()
    plt.imshow((tst_img[0].reshape(32,24)), plt.cm.gray)
    plt.show()
    print(tst_lbl[0])



''' MAIN PROGRAM '''
        
question1()
question2()
question3()
best_classifier()


