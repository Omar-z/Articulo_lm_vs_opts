import numpy as np
import torch
import matplotlib.pyplot as plt

def __R__(X,YH,T):
    tm = T.mean()
    yhm = YH.mean()
    a = ((T-tm)*(YH-yhm)).sum()
    b1 = ((T-tm)**2).sum()
    b2 = ((YH-yhm)**2).sum()
    return a.cpu().detach().numpy()/(np.sqrt(b1.cpu().detach().numpy()*b2.cpu().detach().numpy())+1e-12) #patch división entre 0

def __R2__(YH,T):
    tm = T.mean()
    a = ((T-YH)**2).sum()
    b = ((T-tm)**2).sum()
    return 1-(a.cpu().detach().numpy()/b.cpu().detach().numpy())

def confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int,plot=True,debug=False) -> torch.Tensor:
    pred_labels = y_pred.argmax(dim=1).long()
    y_true = y_true.argmax(dim=1).long()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, pred_labels):
        cm[t, p] += 1
    
    print(cm) if debug else ""
    print("Accuracy: {0:.3f}\nPrecision: {1:.3f}".format(*get_accuracy_precision(cm)))  if debug else ""
    if plot:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('matrix de confusion')
        for i in range(num_classes):
            for j in range(num_classes):
                count = cm[i, j].item()
                plt.text(j, i, str(count),
                        ha='center', va='center',
                        color='white' if count > cm.max()/2 else 'black')
        plt.show()
    return get_accuracy_precision(cm)

def get_accuracy_precision(cm: torch.Tensor):
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    accuracy = correct / total if total else 0

    num_classes = cm.size(0)
    precisions = []
    for i in range(num_classes):
        col_sum = cm[:, i].sum().item()
        if col_sum == 0:
            precisions.append(0.0)
        else:
            precisions.append(cm[i, i].item() / col_sum)
    precision_macro = sum(precisions) / num_classes

    return accuracy, precision_macro 

def PlotTraining(X,YH,T,plot=True,debug=False):
    x = X.cpu().detach().numpy()
    yh = YH.cpu().detach().numpy()
    t = T.cpu().detach().numpy()
    n,_ = x.shape
    _,m = t.shape
    B = (x.T@x)**-1@(x.T@t)
    rl = x@B#A*X+B
    SSe = ((t-yh)**2).flatten().sum()
    R = __R__(X,YH,T)
    R2 =__R2__(YH,T)
    
    if debug:
        print(f"SSE = {SSe:E}")
        print(f"MSE = {(1/n)*SSe:E}")
        print(f"R = {R}")
        print(f"R^2 = {R2}")
    
    if plot:
        fig,ax = plt.subplots(m,2,figsize=(12,8),sharex=True,sharey=True)

        for ren in range(m):
            if m>1:
                ax[ren ,0].plot(t[:,ren],"-",color='black')
                ax[ren, 0].set_title(f"Target_{ren+1}")
                ax[ren ,1].plot(yh[:,ren],"-",color='blue')
                ax[ren, 1].set_title(f"R={R}\n$R^2$={R2}")
            else:
                ax[0].plot(t[:],"-",color='black')
                ax[0].set_title(f"Target_{ren+1}")
                ax[1].plot(yh[:],"-",color='blue')
                ax[1].set_title(f"R={R}\n$R^2$={R2}")
        fig.tight_layout()
        plt.show()
    return R, R2

def OneHotEncode(X,clases=2)->torch.Tensor:
    #print(f"X shape: {X.shape},{clases}")
    return torch.nn.functional.one_hot(X.to(torch.long),clases).squeeze().float()

def getLane(OM: np.ndarray, lane: str) -> float:
    """
    Función que determina a que distancia se encuentra los carros en cada carril con respecto
    a el agente.\n

    El agente se puede encontrar en cualquiera de los 3 carriles\n 
    (0, arriba), (4, medio), (8, abajo)\n

    y regresa las distancias de los carros relativo al agente\n
    
    0: [(d,0,c_arriba),(d,4,c_medio),(d,8,c_abajo)]\n
    4: [(d,-4,c_arriba),(d,0,c_medio),(d,4,c_abajo)]\n
    8: [(d,-8,c_arriba),(d,-4,c_medio),(d,0,c_abajo)]\n

    donde\n 
    
    @d: es la disntancia\n
    @los numeros: es el codigo para identificar el carril\n
    @c_: es los otros carros en el carril\n

    """
    agente_carril_int, agente_carril_str = QueCarrilVoy(OM[0:1,:])
    r, _ = OM.shape

    if(agente_carril_str=="arriba"):
        for i in range(1,r):
            if lane == "middle":
                if np.round(OM[i,1])==4: # hay carro carril arriba #OM[i, 1] >= -4 and OM[i, 1] <= -3.4:
                    return OM[i, 0]
            elif lane == "top":
                if np.round(OM[i,1])>=-0.1 and np.round(OM[i,1])<=0.1:# hay carro en el mismo carril #OM[i, 1] >= -0.5 and OM[i, 1] <= 0.5:
                    return OM[i, 0]
            else:  # bottom
                if np.round(OM[i,1])==8:# hay carro carril abajo #OM[i, 1] <= 4 and OM[i, 1] >= 3.4:
                    return OM[i, 0]

    elif(agente_carril_str=="medio"):
        for i in range(1,r):
            if lane == "top":
                if np.round(OM[i,1])==-4: # hay carro carril arriba #OM[i, 1] >= -4 and OM[i, 1] <= -3.4:
                    return OM[i, 0]
            elif lane == "middle":
                if np.round(OM[i,1])>=-0.1 and np.round(OM[i,1])<=0.1:# hay carro en el mismo carril #OM[i, 1] >= -0.5 and OM[i, 1] <= 0.5:
                    return OM[i, 0]
            else:  # bottom
                if np.round(OM[i,1])==4:# hay carro carril abajo #OM[i, 1] <= 4 and OM[i, 1] >= 3.4:
                    return OM[i, 0]
    elif(agente_carril_str=="abajo"):
            for i in range(1,r):
                if lane == "top":
                    if np.round(OM[i,1])==-8: # hay carro carril arriba #OM[i, 1] >= -4 and OM[i, 1] <= -3.4:
                        return OM[i, 0]
                elif lane == "bottom":
                    if np.round(OM[i,1])>=-0.1 and np.round(OM[i,1])<=0.1:# hay carro en el mismo carril #OM[i, 1] >= -0.5 and OM[i, 1] <= 0.5:
                        return OM[i, 0]
                else:  # middle
                    if np.round(OM[i,1])==-4:# hay carro carril abajo #OM[i, 1] <= 4 and OM[i, 1] >= 3.4:
                        return OM[i, 0]

    return -1  # no importa

def GetActionNumber(raw_num, version=2):
    if(version==2): # es salidas
        if raw_num >2:
            return 1
        return raw_num
    # 5 salidas
    return raw_num

def QueCarrilVoy(obs: np.ndarray,ant =None) -> tuple[int,str]:
    """
    Determina en que carril se encuentra el agente\n
    -1: abajo\n
    0: medio\n
    1: arriba\n

    """
    if obs[0,1] >= 0 and obs[0,1] <= 2:
        return 1,"arriba"
    elif obs[0,1] >=7 and obs[0,1]<= 9:
        return -1,"abajo"
    elif obs[0,1] >=3 and obs[0,1]<= 5:
        return 0,"medio"
    return ant,"no se sabe"

def actionToStr(actions: int) -> str:
    """
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
    """
    if actions == 0:
        return "LANE_LEFT"
    elif actions == 1:
        return "IDLE"
    elif actions == 2:
        return "LANE_RIGHT"
    elif actions == 3:
        return "FASTER"

    return "SLOWER"

#pasarlo a un sistema difuso o un randomforrest
def RetroAlimentacionBaseReglas(estado:list[int,float,float,float],estado_velocidad:int, **args)->int:
    """
    Función que le dice al modelo cual era la acción mas probable de hacer en una instancia determinada
    el peso de las acciones en este caso es lo mismo (1) con excepción de la acción velocidad
    params
    @estado: lista de 4 elementos que representan el estado actual deel mundo (carril_agente, carril_arriba, carril_medio, carril_abajo)
    @estado_velocidad: entero que representa la velocidad del agente(-1,0,1)[lento,normal,rapido]
    @reglas: lista por cada situación de reglas que se deben seguir para tomar una acción
    return  int: la acción que se debió haber tomado
    """
    """
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
    """
    #forzar a moverse de carril
    agente = estado[0]
    if(agente == 0): #medio
        #arriba o abajo sin carros
        a_arriba = estado[1]
        a_abajo = estado[3]
        if(a_arriba <30 and a_abajo >30):
            opciones=[2]
            return np.random.choice(opciones)
        elif(a_arriba >30 and a_abajo <30):
            opciones=[0]
            return np.random.choice(opciones)
        elif(a_arriba >30 and a_abajo >30):
            opciones=[0,2]
            return np.random.choice(opciones)
        else:
            return np.random.choice([0,2]);
    elif(agente == 1): #arriba
        opciones = [2]
        return np.random.choice(opciones)

    elif(agente == -1): #abajo
        opciones = [0]
        return np.random.choice(opciones)

    return 1