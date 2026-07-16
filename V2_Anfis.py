import torch
import torch.nn as nn
from torch.autograd import grad
from torch.func import functional_call, jacfwd
import threading, itertools, time, sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from Anfis_utils import CargarFIS,CrearFISInicial,GuardarFIS
from funciones_auxiliares import OneHotEncode,PlotTraining, confusion_matrix

#clase abstracta para el optimizadores
class Optimizador:
    def step(self):
        raise NotImplementedError("El optimizador no tiene implementado el método")
    def setParams(self,*args):
        raise NotImplementedError("El optimizador no tiene implementado el método")

class CapaGaussiana(nn.Module):
    def __init__(self, n_in, k_reglas, ant:np.array=None):
        """
        @param n_in: número de entradas (N)
        @param k_reglas: número de membresías por cada entrada y reglas (K)
        @param self.centro, self.sigma tienen forma (n_in, k_reglas)
        @param ant = valores de las membresias extraidos del FIS
        """
        super(CapaGaussiana, self).__init__()
        self.n_in = n_in
        self.reglas = k_reglas

        self.centro = nn.Parameter(torch.zeros(n_in, k_reglas).to(torch.float64))
        #evitar division entre cero
        self.sigma = nn.Parameter(torch.zeros(n_in, k_reglas).to(torch.float64) + 0.1)

        if(ant is not None):
            #cargar los valores de las membresias del ant
            #(num_in, num_reglas, 2) por gausiana el 2
            self.__ExtraerMembresias(ant)

            #print(f"sigma: {self.sigma.shape}")
            #print(f"centro: {self.centro.shape}")

    def __ExtraerMembresias(self,datos:np.array)->None:
        # datos: (num_in, num_reglas, 2) -> [...,1]=centro, [...,0]=sigma
        # vectorizado: un solo copy_ por parámetro en vez de N*K asignaciones escalares
        self.centro.data.copy_(torch.as_tensor(datos[..., 1], dtype=torch.float64))
        self.sigma.data.copy_(torch.as_tensor(datos[..., 0], dtype=torch.float64))

    def forward(self, x):
        """
        @param x: (muestras, n_in)
        Salida: (muestras, k_reglas)
          membership_vals[:, i, j] = exp(-0.5*((x_i - mu[i, j]) / sigma[i, j])^2)
        """
        # x.shape => (muestras, n_in)
        # centro.shape => (n_in, k_reglas) => expandimos a (1, n_in, k_reglas)

        x_exp = x.unsqueeze(-1)                     # (muetras, n_in, 1)
        mu_exp = self.centro.unsqueeze(0)           # (1, n_in, k_reglas)
        sigma_exp = self.sigma.unsqueeze(0)         # (1, n_in, k_reglas)

        # Gaussiana
        # (muestras, n_in, k_reglas)
        diff = (x_exp-mu_exp)/sigma_exp
        membership_vals = torch.sum(diff**2,dim=1)  # [muestras, k_reglas]
        #print(f"memb: {membership_vals.shape}")
        return membership_vals                      # (muestras, k_reglas)

class CapaFuerzaDisparo(nn.Module):
    def __init__(self, n_in, k_reglas,muestras=719):
        """
        @param n_in: N
        @param k_reglas: K
        la combinacion de las reglas es lineal ya que por
        cada entrada es una reglla ejemplo.
        1 1 1
        2 2 2
        3 3 3
        por lo que podemos hacer el producto como la fuerza de disparo y por la
        propiedad de los exponenciales seria la sumatoria (recibe por parametro)
        """
        super(CapaFuerzaDisparo, self).__init__()
        self.n_in = n_in
        self.k_reglas = k_reglas #reglas


    def forward(self, membership_vals):
        """
        membership_vals: (muestras,k_reglas)
        Retorna un tensor (muestras, k_rules) 
        donde por cada entrada existe una regla
        1 1 1 1
        2 2 2 2
        3 3 3 3 
        """

        rule_activations = torch.exp(-0.5*membership_vals)     # (muestras, k_reglas)
        #print(f"fd: {ALPHA.shape}")
        #rule_activations = ALPHA
        return rule_activations                     # (muestras, k_reglas)


class NormalLayer(nn.Module):
    def __init__(self):
        """
        Normaliza la fuerza de dispparo
        alpha/sum(alpha)
        """
        super(NormalLayer, self).__init__()

    def forward(self, rule_activations):
        phi = rule_activations/torch.sum(rule_activations,dim=1,keepdim=True)

        return phi                          # (muestras, reglas)


class CapaCenterOfSets(nn.Module):
    def __init__(self, num_rules, n_out, con:np.array=None):
        """
        @param num_rules 
        @param n_out 
        @param con = valores de los consecuentes extraidos del FIS
        """
        super(CapaCenterOfSets, self).__init__()
        self.num_rules = num_rules
        self.n_out = n_out

        # (num_rules, n_out)
        self.centers = nn.Parameter(torch.zeros(num_rules, n_out).to(torch.float64))
        if(con is not None):
            #cargar las membresias del fis (num_out, num_reglas, 1) ya que es constante
            self.__ExtraerMembresias(con)
            #print(f"theta: {self.centers.shape}")

    def __ExtraerMembresias(self,data:np.array) -> None:
        # data: (num_out, num_reglas, 1); centers: (num_rules, n_out) -> transponer
        self.centers.data.copy_(torch.as_tensor(data[..., 0], dtype=torch.float64).T)

    def forward(self, rule_activations):
        """
        rule_activations: (muestras, num_rules)
        Retorna: (muestras, n_out)
        y = fuerza_normalizada . constantes(centros)
        """

        #centers_exp = self.centers.unsqueeze(0)             # (1, num_rules, n_out)
        #print(self.centers.shape,"@",rule_activations.shape)
        output = rule_activations @ self.centers
        return output


class RLANFISBuilder:
    def __init__(self):
        self.anfis = None
        self.fis = None
        self.ins = None
        self.out = None
        self.reglas = None
        self.mu_inc = 10
        self.mu_dec = 0.1
        self.mu_max = 1e10
        self.valmaxfails = 20
        self.tipo = "regresion"
        self.optimizador = None

    
    def AddFIS(self, fis:str):
        self.fis = fis
        return self

    def AddInputs(self, i:int):
        self.ins = i
        return self
    
    def AddOutputs(self, o:int):
        self.out = o
        return self
    
    def AddRules(self, r:int):
        self.reglas = r
        return self

    def AddMuStats(self, mu_dec:float, mu_inc:float, mu_max:float):
        self.mu_dec = mu_dec
        self.mu_inc = mu_inc
        self.mu_max = mu_max
        return self
    
    def AddValMaxFails(self, valmaxfails:int):
        self.valmaxfails = valmaxfails
        return self 
    
    def AddTipoProblema(self,tipo:str):
        self.tipo = tipo
        return self
    
    def AddOptimizador(self, optimizador:type, **kwargs):
       #if not (isinstance(optimizador, type) and issubclass(optimizador, Optimizador)):
            #raise TypeError("optimizador_cls debe ser una subclase de Optimizador")
        self.optimizador = optimizador
        self.opt_args = kwargs
        return self
    
    def AddFunctionLoss(self, fn_loss):
        self.fn_loss = fn_loss
        return self
    
    def Build(self):
        if(self.fis==None or self.ins==None or self.out==None or self.reglas==None):
            raise Exception("Para construir el modelo se ocupa como minimo el fis, numero de entradas, salidas y reglas")

        self.anfis = ANFISND(self.ins,self.out,self.reglas,self.fis,
                            #self.mu_inc,self.mu_dec,self.mu_max,
                            self.valmaxfails,
                            self.tipo,
                            self.fn_loss)
                            #self.optimizador,
                            #self.opt_args)
    
        if not (isinstance(self.optimizador, type) and issubclass(self.optimizador, Optimizador)):
            self.anfis.optimizador = self.optimizador(self.anfis.parameters(), **self.opt_args)
        else:
            self.anfis.optimizador = self.optimizador(self.anfis, **self.opt_args)
        #print(self.anfis.optimizador)
        #print(self.fn_loss)
        return self.anfis

class ANFISND(nn.Module):
    def __init__(self, n_in, n_out, k_reglas, fis_path:str=None, 
                valmaxfails=20,tipo:str="regresion",#muinc=10,mudec=0.1,mumax=1e50,valmaxfails=20,tipo:str="regresion",
                fn_loss=None):      #optimizador:Optimizador=None):
        """
        n_in: N
        n_out: M
        k_reglas: K
        """
        super(ANFISND, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.k_reglas = k_reglas
        #pal optimizador
        #self.mu=0.01
        #self.mu_inc = muinc
        #self.mu_dec = mudec
        #self.mu_max = mumax
        self.num_fallos=0
        self.num_max_fallos = valmaxfails
        self.fis =None
        self.tipo = tipo
        self.fn_loss = fn_loss
        
        a_mem=None
        c_mem=None
        if(fis_path is not None):
            # regresa una tupla (fis, (num_in,num_reglas,2), (num_out, num_reglas,1))
            fis,a_mem,c_mem = CargarFIS(fis_path)

        self.membership_layer = CapaGaussiana(n_in, k_reglas, a_mem)
        self.rule_layer = CapaFuerzaDisparo(n_in, k_reglas)
        self.normal_layer = NormalLayer()

        num_rules = k_reglas 
        self.defuzz_layer = CapaCenterOfSets(num_rules, n_out, c_mem)
        
        #self.optimizador = optimizador #optimizador(self)
        #self.optimizador.setParams(0.01,self.mu_dec,self.mu_inc,self.mu_max)
        

    def forward(self, x):
        """
        x: (muestras, n_in)
        """
        membership_vals = self.membership_layer(x)          # (muestras, k_reglas)
        #print(f"[{membership_vals.shape}]")
        rule_acts = self.rule_layer(membership_vals)        # (muestras, k_reglas)
        #print(f"[{rule_acts.shape}]")
        norm_acts = self.normal_layer(rule_acts)            # (muestras, k_reglas)
        #print(f"[{norm_acts.shape}]")
        y = self.defuzz_layer(norm_acts)                    # (batch_size, n_out)
        #print(f"[{y.shape}]")
        #return y
        # en clasificación no tener valores negativos en la salida 
        # ya que representan la probabilidad de que sea el objeto
        return y #torch.nn.functional.softmax(y,dim=1) if self.tipo=="clasificacion" else y


class LevenberMaquardtOpt(Optimizador):
    def __init__(self,model,lambda_init=0.01,lambda_decr=0.9,lambda_incr=10, device=torch.device("cpu"))->None:
        self.model = model
        self.lambda_val = lambda_init
        self.lambda_decr = lambda_decr
        self.lambda_incr = lambda_incr
        self.lambda_max = 1e10
        self.nombre = "LM"
        self.device = device
        
        self.params = list(model.parameters())
        self.num_params = sum(p.numel() for p in self.params)
    
    def setParams(self, *args):
        #super().setParams(*args)
        #print(args)
        self.lambda_val = args[0]
        self.lambda_decr = args[1]
        self.lambda_incr = args[2]
        self.lambda_max = args[3]
    
    def _get_param_vector(self)->torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.params])
    
    def _set_param_vector(self, vector:torch.Tensor)->None:
        id = 0
        for param in self.params:
            param_size = param.numel()
            param.data = vector[id:id+param_size].view_as(param.data)
            id+= param_size
    
    def _compute_error(self, Y_pred:torch.Tensor, Y_true:torch.Tensor)->torch.Tensor:
        error_vec=[]
        for t,y in zip(Y_true,Y_pred):
            error = t-y
            error_vec.append(error.view(-1))
        return torch.cat(error_vec).to(torch.float64)
    
    
    
    def jacobiana(self,X,Y) -> tuple[torch.Tensor,torch.Tensor]:
        # Jacobiano exacto del residuo r(theta) = (Y - modelo(X)) por autodiff
        # en modo forward (jacfwd). Reemplaza el bucle de num_params forward-passes
        # con diferencias finitas: una sola llamada vectorizada (vmap) -> GPU
        # y sin error de epsilon. jacfwd conviene aqui porque error_size >> num_params.
        nombres = [n for n, _ in self.model.named_parameters()]
        formas  = [p.shape for _, p in self.model.named_parameters()]
        numels  = [p.numel() for _, p in self.model.named_parameters()]

        # theta0: mismo orden/aplanado (row-major) que _get_/_set_param_vector,
        # asi las columnas del Jacobiano casan con el vector de parametros.
        theta0 = self._get_param_vector().detach()

        def residuo(theta):
            p, i = {}, 0
            for nombre, forma, n in zip(nombres, formas, numels):
                p[nombre] = theta[i:i+n].view(forma)
                i += n
            out = functional_call(self.model, p, (X,))
            return (Y - out).reshape(-1)          # error = Y_true - Y_pred

        curr_error = residuo(theta0).detach()
        jacob = jacfwd(residuo)(theta0)           # (error_size, num_params)

        return jacob, curr_error
    
    def step(self,X,Y):
        jacobiana, error_vec = self.jacobiana(X,Y)
        #print(f"jacobiana {jacobiana.is_cuda}, error: {error_vec.is_cuda}")
        # dispositivo/dtype se toman del Jacobiano (= donde viven los parametros)
        # para que todo el paso corra en GPU sin mezclar CPU/GPU.
        #JTJ
        JtJ = torch.matmul(jacobiana.t(), jacobiana)
        diag_JtJ = torch.eye(jacobiana.shape[1], dtype=jacobiana.dtype, device=jacobiana.device)
        g = torch.matmul(jacobiana.t(), error_vec)
        
        #parametros
        c_params = self._get_param_vector()
        
        #JJ = JtJ.clone()
        #JJ[I,I] += self.lambda_val
        A = JtJ + self.lambda_val * diag_JtJ
        try:
            #(JtJ+lambdaDiag)^-1 = Je
            delta = -torch.linalg.solve(A, g)
        except:
            delta = -torch.zeros_like(g)

        #actualizar params
        #print(f"delta: {delta.shape} c_params: {c_params.shape}")
        n_params = c_params +delta
        self._set_param_vector(vector=n_params)
        #self._set_param_vector(delta)
        
        n_out = self.model(X)
        n_error = self._compute_error(n_out,Y)
        
        c_loss = torch.sum(error_vec**2)
        n_loss = torch.sum(n_error**2)
        
        if(n_loss < c_loss):
            self.lambda_val *= self.lambda_decr
            return n_loss.item()
        else:
            self.lambda_val *= self.lambda_incr
            self._set_param_vector(c_params)
            return c_loss.item()
    
    


def train_nfs(model, X_train, y_train, epochs=100,tolerancia=1e-6, debug=False, fn_loss_lst:dict={},
              early_stop={"fallos_init":0.01,"fallos_inc":10,"fallos_dec":0.1,"fallos_tol":1e20}) -> tuple[list[float],dict ]:
    """
    Train the neuro-fuzzy system using Levenberg-Marquardt optimization
    """
    print(f"Inputs: {X_train.shape} , Outputs: {y_train.shape}") if debug else ""
    optimizer = model.optimizador 
    fn_loss = model.fn_loss
    #LevenberMaquardtOpt(model,
                                #  model.mu,
                                #  model.mu_dec,
                                # model.mu_inc)
    losses = []
    
    estado={
        "iter_act":0,
        "iter_total":epochs,
        "flair": f"loss \n",
        "color": "\033[1;45;36m",
        "cursor":"\033[3;1H"
    }
    
    fn_metricas=fn_loss_lst
    metricas ={}
    for fn_name, fn in fn_metricas.items():
        metricas[fn_name] = []
    
    stop_event = threading.Event()
    barra = threading.Thread(target=mostrar_barra_progreso,args=(estado,stop_event),daemon=True)
    barra.start()

    #parado antes
    mejor_loss = float("inf")
    fallos_init =early_stop["fallos_init"]
    fallos_inc = early_stop["fallos_inc"]
    fallos_dec = early_stop["fallos_dec"]
    fallos_tol = early_stop["fallos_tol"]
    min_delta = 1e-5
    
    for epoch in range(epochs):
        #optimizadores de pytorch no usan parámetros en el step
        #if not (isinstance(optimizer, type) and issubclass(optimizer, LevenberMaquardtOpt)):
        out = model(X_train)
        if(getattr(optimizer,"nombre",None) != "LM"):
            optimizer.zero_grad()
            #out = model(X_train)
            loss = fn_loss(out, y_train)
            loss.backward()
            optimizer.step()
        else: 
            loss = optimizer.step(X_train, y_train)
        
        
        for fn_name, fn in fn_metricas.items():    
            metricas[fn_name].append(fn(out, y_train).item())
        
        
        estado["iter_act"]=epoch
        if isinstance(loss, torch.Tensor):
            estado["flair"] = f"loss: {loss.item():.8f}\n"
            losses.append(loss.item())  
        else:
            estado["flair"] = f"loss: {loss:.8f}\n"
            losses.append(loss)

        
        if getattr(optimizer, "nombre",None) !="LM":
            if loss < mejor_loss - min_delta:
                mejor_loss = loss
                fallos_init *= fallos_dec
            else:
                fallos_init *= fallos_inc
            
            if fallos_init >= fallos_tol:
                print(f"[{epoch+1}] El modelo llego a fallas maximas {fallos_init:2d} >= {fallos_tol:2d}") if debug else ""
                print(f"[{epoch+1}] con un loss de {loss:.6f}") if debug else ""
                stop_event.set()
                return losses, metricas
        
        if (epoch % int(epochs*.1) if epochs >100 else 10) == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}") if debug else ""
        
        if(loss <= tolerancia):
            print(f"Se llego a la tolerancia {loss:.6f} <= {tolerancia}") if debug else ""
            stop_event.set()
            return losses, metricas
        
        if(getattr(optimizer,"nombre",None) == "LM"):
            if(optimizer.lambda_val > optimizer.lambda_max):
                print(f"[{epoch+1}] El modelo llego a las mu maximas {optimizer.lambda_val:.1E} >= {optimizer.lambda_max:.1E}({optimizer.lambda_val>=optimizer.lambda_max})") if debug else ""
                print(f"[{epoch+1}] con un loss de {loss:.6f}") if debug else ""
                stop_event.set()
                return losses, metricas
    
    stop_event.set()
    barra.join()

    return losses, metricas


def train_nfs_batch(model, X_train, y_train, epochs=100, batch_size=32, tolerancia=1e-6, shuffle=True, debug=False, fn_loss_lst:dict={},
                    early_stop={"fallos_init":0.01,"fallos_inc":10,"fallos_dec":0.1,"fallos_tol":1e20}) -> tuple[list[float],dict ]:
    """
    Train the neuro-fuzzy system using mini-batch optimization
    """
    print(f"Inputs: {X_train.shape} , Outputs: {y_train.shape}") if debug else ""
    optimizer = model.optimizador
    fn_loss = model.fn_loss
    losses = []

    estado={
        "iter_act":0,
        "iter_total":epochs,
        "flair": f"loss \n",
        "color": "\033[1;45;36m",
        "cursor":"\033[3;1H"
    }

    fn_metricas=fn_loss_lst
    metricas ={}
    for fn_name, fn in fn_metricas.items():
        metricas[fn_name] = []

    n_muestras = X_train.shape[0]
    #dispositivo donde viven los parametros; los datos se quedan en el host
    device = next(model.parameters()).device

    stop_event = threading.Event()
    barra = threading.Thread(target=mostrar_barra_progreso,args=(estado,stop_event),daemon=True)
    barra.start()
    
    #parado antes
    mejor_loss = float("inf")
    fallos_init =early_stop["fallos_init"]
    fallos_inc = early_stop["fallos_inc"]
    fallos_dec = early_stop["fallos_dec"]
    fallos_tol = early_stop["fallos_tol"]
    min_delta = 1e-5
    
    for epoch in range(epochs):
        #se recorren los datos en un orden distinto cada epoca
        indices = torch.randperm(n_muestras) if shuffle else torch.arange(n_muestras)

        loss_acum = 0.0
        met_acum = {fn_name: 0.0 for fn_name in fn_metricas}
        for inicio in range(0, n_muestras, batch_size):
            batch_idx = indices[inicio:inicio+batch_size]
            #solo el batch actual se transfiere a device
            X_batch = X_train[batch_idx].to(device)
            y_batch = y_train[batch_idx].to(device)
            b = X_batch.shape[0]

            out = model(X_batch)
            if(getattr(optimizer,"nombre",None) != "LM"):
                optimizer.zero_grad()
                loss = fn_loss(out, y_batch)
                loss.backward()
                optimizer.step()
            else:
                loss = optimizer.step(X_batch, y_batch)

            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            #se ponderan loss y metricas por el tamaño real del batch
            loss_acum += loss_val * b
            for fn_name, fn in fn_metricas.items():
                met_acum[fn_name] += fn(out, y_batch).item() * b

            #se descarta el batch de device antes de la siguiente iteracion
            del X_batch, y_batch, out, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        #promedios de la epoca sobre todas las muestras
        loss_epoch = loss_acum / n_muestras
        losses.append(loss_epoch)
        for fn_name in fn_metricas:
            metricas[fn_name].append(met_acum[fn_name] / n_muestras)

        estado["iter_act"]=epoch
        estado["flair"] = f"loss: {loss_epoch:.8f}\n"

        if (epoch % int(epochs*.1) if epochs >100 else 10) == 0:
            print(f"Epoch {epoch}, Loss: {loss_epoch:.6f}") if debug else ""

        if getattr(optimizer, "nombre",None) !="LM":
            if loss_epoch < mejor_loss - min_delta:
                mejor_loss = loss_epoch
                fallos_init *= fallos_dec
            else:
                fallos_init *= fallos_inc
            
            if fallos_init >= fallos_tol:
                print(f"[{epoch+1}] El modelo llego a fallas maximas {fallos_init:2d} >= {fallos_tol:2d}") if debug else ""
                print(f"[{epoch+1}] con un loss de {loss_epoch:.6f}") if debug else ""
                stop_event.set()
                return losses, metricas

        if(loss_epoch <= tolerancia):
            print(f"Se llego a la tolerancia {loss_epoch:.6f} <= {tolerancia}") if debug else ""
            stop_event.set()
            return losses, metricas

        if(getattr(optimizer,"nombre",None) == "LM"):
            if(optimizer.lambda_val > optimizer.lambda_max):
                print(f"[{epoch+1}] El modelo llego a las mu maximas {optimizer.lambda_val:.1E} >= {optimizer.lambda_max:.1E}({optimizer.lambda_val>=optimizer.lambda_max})") if debug else ""
                print(f"[{epoch+1}] con un loss de {loss_epoch:.6f}") if debug else ""
                stop_event.set()
                return losses, metricas

    stop_event.set()
    barra.join()

    return losses, metricas


def CapaCompetitiva(X)->torch.Tensor:
    indices = X.argmax(dim=1)
    return torch.nn.functional.one_hot(indices,num_classes=X.shape[1])


def gx_train(modelo:nn.Module, loss:torch.Tensor,YH:torch.Tensor,T:torch.Tensor,X_VAL:torch.Tensor,Y_VAL:torch.Tensor,learning_rate=1) -> None:
    grads = grad(loss, modelo.parameters(), create_graph=True)

    with torch.no_grad():
        for p in modelo.parameters():
            numel = p.numel()
            update = p.view(p.shape) #delta[idx: idx+numel].view(p.shape)
            #print(f"update shape: {update.shape}")
            p -= learning_rate * update
            idx += numel

def lm_train(modelo:nn.Module, loss:torch.Tensor,YH:torch.Tensor,T:torch.Tensor,X_VAL:torch.Tensor,Y_VAL:torch.Tensor,learning_rate=1,tipo="regresion") -> None:
    error_vec = (YH - T).view(-1)
    error_vec = error_vec.to(torch.float64)

    J_rows = []
    for i in range(len(error_vec)):
        e_i = error_vec[i]
        grad_i = grad(e_i, modelo.parameters(), retain_graph=True)
        row_i = []
        for gi in grad_i:
            row_i.append(gi.view(-1))
        row_i = torch.cat(row_i)
        #print(f"\033[33mrow_i: {row_i.shape}\033[0m")
        J_rows.append(row_i.unsqueeze(0))

    J = torch.cat(J_rows, dim=0)  
    # A = (J^T J + λ diag(J^T J)), g = J^T error
    JTJ = J.t() @ J
    diag_JTJ = torch.eye(J.shape[1])
    gxNorm = torch.norm(2*J.t()@error_vec)

    while(modelo.mu <= modelo.mu_max):
        A = JTJ + modelo.mu * diag_JTJ
        #print(f"{J.t().shape} @ {error_vec.shape}")
        g = J.t() @ error_vec
        try:
            delta = -torch.linalg.solve(A, g)
        except RuntimeError:
            delta = -torch.zeros_like(g)


        ytest = modelo(X_VAL)
        if(tipo=="clasificacion"):
            #ytest = CapaCompetitiva(ytest)
            n = T.shape[0]
            clamp_probs = torch.clamp(ytest,min=1e-8,max=1.0-1e-8)
            perf1 = -(1/n)*torch.sum(Y_VAL*torch.log(clamp_probs))
        else:
            perf1 =torch.sum((ytest - Y_VAL)**2)
        # Actualizamos parámetros
        idx = 0
        estado_temporal= copy.deepcopy(modelo)
        with torch.no_grad():
            for p in modelo.parameters():
                numel = p.numel()
                update = delta[idx: idx+numel].view(p.shape)
                #print(f"update shape: {update.shape}")
                p += learning_rate * update
                idx += numel
        
        #cambiar a validacion
        ytest = modelo(X_VAL)
        if(tipo=="clasificacion"):
            #ytest = CapaCompetitiva(ytest)
            clamp_probs = torch.clamp(ytest,min=1e-8,max=1.0-1e-8)
            perf2 = -(1/n)*torch.sum(Y_VAL*torch.log(clamp_probs))
        else:
            perf2 =torch.sum((ytest - Y_VAL)**2)
        #print(f"[{loss.item()} > {etes.item()}]")
        if(perf2.item()< perf1.item()):
            if(modelo.mu>1e-300): # por si llega al num maximo en python
                modelo.mu*=modelo.mu_dec
            return #break
        modelo.load_state_dict(estado_temporal.state_dict())
        modelo.mu*=modelo.mu_inc
        #print(f"[{modelo.mu:.1e}] incrementar MU")
        #fin de while
    #fin de lm_train

# -------------------------------------------------------------------


def cantidad_reglas(entradas:int,salidas:int,n_clases:int,
                    r_inicial:int,r_final:int,
                    x_train:np.ndarray, y_train:np.ndarray,
                    x_test:np.ndarray, y_test:np.ndarray,
                    debug=False):
    X = torch.from_numpy(x_train).float()
    Y = torch.from_numpy(y_train).float()
    XTEST = torch.from_numpy(x_test).float()
    YTEST = torch.from_numpy(y_test).float()
    
    log = []
    best_regla = 0
    best_acc = 1e-20 if n_clases!=None else 1e20
    
    if n_clases !=None:
        Y = OneHotEncode(Y,n_clases)
        YTEST = OneHotEncode(YTEST,n_clases)
    
    estado = {
        "iter_act":r_inicial,
        "iter_total":r_final+1,
        "color": "\033[0;34m",
        "cursor":"\r"
    }
    stop_event = threading.Event()
    barra = threading.Thread(target=mostrar_barra_progreso,args=(estado,stop_event),daemon=True)
    barra.start()
    for i in range(r_inicial,r_final+1):
        regla = i
        
        estado["iter_act"]=regla
        
        
        #creamos el fis
        fis, fis_str = CrearFISInicial("tempFIS",
                                        pd.DataFrame(x_train),
                                        pd.DataFrame(y_train.squeeze()) if n_clases==None else pd.DataFrame(Y.detach().numpy()),
                                        regla)
        
        modelo = RLANFISBuilder()\
                .AddFIS(fis_str)\
                .AddInputs(entradas)\
                .AddOutputs(salidas)\
                .AddRules(regla)\
                .AddValMaxFails(20)\
                .AddOptimizador(LevenberMaquardtOpt, lambda_init=0.01,lambda_decr=0.9,lambda_incr=10)\
                .AddFunctionLoss(torch.nn.MSELoss())\
                .Build()
        
        loss = train_nfs(modelo,
                        X,Y,
                        epochs=1000,
                        tolerancia=1e-12,
                        debug=debug)
        
        if np.isnan(loss).any(): continue
        
        y_hat = modelo(XTEST)
        
        if n_clases != None:
            acc,prec = confusion_matrix(y_hat,
                                        YTEST,
                                        n_clases,
                                        plot=False,)
            log.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_regla = regla
        else:
            loss = torch.nn.MSELoss()(y_hat,YTEST).item()
            log.append(loss)
            if loss < best_acc:
                best_acc = loss
                best_regla = regla
                
        
    
    stop_event.set()
    barra.join()
        
    return best_regla, log
#

def mostrar_barra_progreso(estado:dict, stop_event)->None:
    frames = itertools.cycle(["|", "/", "—", "\\"])
    cursor = estado["cursor"]
    while not stop_event.is_set():
        iter_act = estado["iter_act"]
        iter_total = estado["iter_total"]
        color = estado["color"]
        frame = next(frames)
        reset = "\033[0m"
        w=40
        porcentaje = "{0:0.1f}".format(100*(iter_act)/(iter_total))
        curr_width = int(w*iter_act//(iter_total))
        barra = color+"█" * curr_width + reset+"░" * (w - curr_width-1)
        sys.stdout.write(cursor+f"{frame}[{iter_act}/{iter_total}][{barra}]{porcentaje}%| {estado["flair"]} ")
        sys.stdout.flush()
        if stop_event.wait(0.1):
            break
        #time.sleep(0.05)
        #if (iter_act+1) == iter_total:
    #sys.stdout.write("\033[J") 
    print("\n",flush=True)
    


if __name__ == "__main__":
    import numpy as np
    from scipy.io import loadmat
    from sklearn.model_selection import train_test_split

    #if sys.argsv[1] == "":
    #np.random.seed(42)
    #torch.manual_seed(42)

    #regresion
    engine_in = loadmat("data_sets/engineInputs.mat")
    engine_tar = loadmat("data_sets/engineTargets.mat")
    simplefit_data = loadmat("data_sets/simple_fit.mat")
    london_weather = pd.read_csv("data_sets/weather_prediction_london/london_weather_clean.csv")

    #clasificacion
    highway_datos = pd.read_csv("manualDatasetHighWay_V3_4in5out_.csv")
    iris_datos = pd.read_csv("../../datos/data_sets/iris/iris.data",header=None)

    engine_in = np.array(engine_in["engineInputs"])
    engine_tar = np.array(engine_tar["engineTargets"])
    simplefit_int = np.array(simplefit_data['inputs'])
    simplefit_out = np.array(simplefit_data['targets'])
    london_in = london_weather.drop(columns=["date","min_temp","max_temp","mean_temp"]).to_numpy()
    london_out = london_weather[["min_temp","max_temp"]].to_numpy()

    highway_in = highway_datos.drop(columns=["mi_accion"]).to_numpy()
    highway_out = highway_datos[["mi_accion"]].to_numpy()

    iris_in  = iris_datos.drop(columns=[4]).to_numpy()
    iris_out = iris_datos[4].map({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}).to_numpy()
    
    engien  = True
    simple  = False and (not engien)
    london  = False and (not engien and not simple)
    highway = False  and (not london and not engien and not simple)
    iris    = False and (not london and not engien and not simple and not highway) 

    test_size = 0.4
    val_size = test_size/2
    clasificacion = False
    num_clases = None

    #para validar hay que sacar primero el set de entrenamiento
    #luego del set de prueba que salio, divirlo en prueba y validacion mita y mita

    if engien and (not simple and not london):
        X_train, X_val, y_train, y_val = train_test_split(engine_in.T, engine_tar.T, test_size=test_size,) #random_state=9)
        XTEST, XVAL, YTEST, YVAL = train_test_split(X_val, y_val,test_size=val_size)
    elif simple and (not engien and not london):
        X_train, X_val, y_train, y_val = train_test_split(simplefit_int.T, simplefit_out.T, test_size=test_size,) #random_state=9)
        XTEST, XVAL, YTEST, YVAL = train_test_split(X_val, y_val,test_size=val_size)
    elif london and (not engien and not simple):
        X_train, X_val, y_train, y_val = train_test_split(london_in, london_out, test_size=test_size,) #random_state=9)
        XTEST, XVAL, YTEST, YVAL = train_test_split(X_val, y_val,test_size=val_size)
    elif highway and (not london and not engien and not simple):
        X_train, X_val, y_train, y_val = train_test_split(highway_in, highway_out, test_size=test_size,) #random_state=9)
        XTEST, XVAL, YTEST, YVAL = train_test_split(X_val, y_val,test_size=val_size)
    elif iris and (not london and not engien and not simple and not highway):
        X_train, X_val, y_train, y_val = train_test_split(iris_in, iris_out, test_size=test_size,)
        XTEST, XVAL, YTEST, YVAL = train_test_split(X_val, y_val,test_size=val_size)

    # Digamos que N=3 entradas, M=2 salidas, K=2 membresías x entrada.
    if engien and (not simple and not london):
        #   #engine   #simple,  #sintetico
        n_in =   2     #1        #3
        n_out =  2     #1        #2
        k_reglas = 3     #3        #2
        fis,fis_str  = CrearFISInicial("engine", pd.DataFrame(engine_in.T), pd.DataFrame(engine_tar.T), k_reglas)
    elif simple and (not engien and not london):
        #   #engine   #simple,  #sintetico
        n_in =   1     #1        #3
        n_out =  1     #1        #2
        k_reglas = 5     #3        #2
        fis,fis_str = CrearFISInicial("simplefit", pd.DataFrame(simplefit_int.T), pd.DataFrame(simplefit_out.T), k_reglas)
    elif london and (not engien and not simple):
        n_in =   6     #1        #3
        n_out =  2     #1        #2
        k_reglas = 6     #3        #2
        fis,fis_str = CrearFISInicial("london", pd.DataFrame(london_in), pd.DataFrame(london_out), k_reglas)
    elif highway and (not london and not engien and not simple):
        n_in =   5    #1        #3
        n_out =  5     #1        #2
        k_reglas = 5    #4        #2
        num_clases=5
        clasificacion=True
    elif iris and (not london and not engien and not simple and not highway):
        n_in =   4
        n_out =  3
        k_reglas = 3
        num_clases=3
        clasificacion=True



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_torch = torch.from_numpy(X_train).to(torch.float64)#.to(device)#torch.from_numpy(X_np)
    X_test = torch.from_numpy(XTEST).to(torch.float64)#.to(device)#torch.from_numpy(X_np)

    X_VAL = torch.from_numpy(XVAL).to(torch.float64)
    Y_VAL = torch.from_numpy(YVAL).to(torch.float64)


    Y_torch = torch.from_numpy(y_train).to(torch.float64)#.to(device)#torch.from_numpy(Y_np)
    Y_test = torch.from_numpy(YTEST).to(torch.float64)#.to(device)#torch.from_numpy(X_np)
    if clasificacion and highway:
        Y_torch = OneHotEncode(Y_torch,num_clases)
        Y_test =  OneHotEncode(Y_test,num_clases)
        Y_VAL =  OneHotEncode(Y_VAL,num_clases)
        fis,fis_str= CrearFISInicial("highway", pd.DataFrame(X_train), pd.DataFrame(Y_torch.numpy().squeeze()), k_reglas)
    elif not clasificacion and highway:
        fis,fis_str= CrearFISInicial("highway", pd.DataFrame(X_train), pd.DataFrame(Y_torch.numpy().squeeze()), k_reglas)

    if clasificacion and iris:
        Y_torch = OneHotEncode(Y_torch,num_clases)
        Y_test =  OneHotEncode(Y_test,num_clases)
        Y_VAL = OneHotEncode(Y_VAL,num_clases)  
        fis,fis_str= CrearFISInicial("iris", pd.DataFrame(X_train), pd.DataFrame(Y_torch.numpy().squeeze()), k_reglas)
    elif not clasificacion and iris:
        fis,fis_str= CrearFISInicial("iris", pd.DataFrame(X_train), pd.DataFrame(Y_torch.numpy().squeeze()), k_reglas)

    # Crear modelo ANFISND
    anfis_model =   RLANFISBuilder() \
                    .AddFIS(fis_str) \
                    .AddInputs(n_in) \
                    .AddOutputs(n_out) \
                    .AddRules(k_reglas) \
                    .AddMuStats(0.1,10,1e20) \
                    .AddValMaxFails(20) \
                    .AddTipoProblema("clasificacion" if clasificacion else "regresion") \
                    .AddOptimizador(LevenberMaquardtOpt) \
                    .Build()

    #anfis_model = anfis_model.to(device) # si device es GPU se puede procesar ahi el feedforward

    # Entrenar pasarlo a un patron de facade ( se lee fassad) para evitar la separacion 
    if(not clasificacion):
        #anfis_model.fit(X_torch, Y_torch, X_VAL, Y_VAL, max_epochs=5000, lambda_lm=0.001, learning_rate=1.0,tol=1e-8)
        train_nfs(anfis_model, X_torch, Y_torch, epochs=1000,tolerancia=1e-12)
    else:
        #anfis_model.fit_clasificacion(X_torch, Y_torch, X_VAL, Y_VAL, max_epochs=1000, lambda_lm=0.001, learning_rate=1.0,tol=1e-8)
        #anfis_model.TestFuncOpt(X_torch, Y_torch, X_VAL, Y_VAL, max_epochs=5000, lambda_lm=0.001, learning_rate=1.0,tol=1e-8)
        train_nfs(anfis_model, X_torch, Y_torch, epochs=1000,tolerancia=1e-12)
        
    fnom = fis_str.split("__")[0]
    GuardarFIS(fnom+"__final.fis",fis=fis,modelo=anfis_model)

    y_hat = anfis_model(X_test)
    if clasificacion:
        confusion_matrix(y_hat,Y_test,num_classes=num_clases,plot=True)
    else:
        PlotTraining(X_test,y_hat,Y_test)

