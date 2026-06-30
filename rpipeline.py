"""
Script para generar los resultados de LM vs todos los optimizadores
por dataset, y luego generar las tablas de resultados, y los gráficos de resultados.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Any
import threading


_timer_inicio = 0.0
timer = 0.0

def timer_start():
    """Inicializa el tiempo en el que empieza a ejecutar el bloque."""
    global _timer_inicio
    _timer_inicio = time.perf_counter()

def timer_end():
    """Finaliza el conteo y guarda el tiempo transcurrido en la variable global 'timer'."""
    global timer
    timer = time.perf_counter() - _timer_inicio
    return timer

def formato_legible(segundos):
    """Convierte una cantidad de segundos a un texto legible, p. ej. '3 horas, 40 minutos, 15 segundos'."""
    segundos = int(segundos)
    horas, resto = divmod(segundos, 3600)
    minutos, segs = divmod(resto, 60)

    partes = []
    if horas:
        partes.append(f"{horas} hora" + ("s" if horas != 1 else ""))
    if minutos:
        partes.append(f"{minutos} minuto" + ("s" if minutos != 1 else ""))
    if segs or not partes:
        partes.append(f"{segs} segundo" + ("s" if segs != 1 else ""))

    return ", ".join(partes)

# ANFIS
from Anfis_utils import CargarFIS,CrearFISInicial,GuardarFIS
from V2_Anfis import RLANFISBuilder,train_nfs,LevenberMaquardtOpt,Optimizador,ANFISND, cantidad_reglas,mostrar_barra_progreso
from funciones_auxiliares import OneHotEncode,PlotTraining, confusion_matrix


@dataclass
class DataOptimizador:
    nombre:str
    params:dict[str,any]
    
    def __repr__(self):
        return f"{self.nombre}({self.params})"

@dataclass
class DataExperimento:
    dataset_path:str
    dataset_header:int
    dataset_target_col:int
    dataset_map_col:dict
    dataset_entradas:int
    dataset_salidas:int
    resultados_path:str
    corridas:int
    epocas:int
    tolerancia:float
    fallos:int
    funcion_perdida: str
    tipo:str
    reglas_inicial:int
    reglas_total:int
    train_size:float
    test_size:float
    val_size:float

@dataclass
class DataConfig:
    optimizadores: list[DataOptimizador] = field(default_factory=list)
    experimentos: DataExperimento = field(default_factory=DataExperimento)


def parsear_json(jsonfile)->DataConfig:
    
    optimizadores = [
        DataOptimizador(nombre=o["name"],params=o["params"])
        for o in jsonfile["optimizadores"]
    ]
    
    experimentos = DataExperimento(**jsonfile["experimentacion"])
    
    return DataConfig(optimizadores=optimizadores,experimentos=experimentos)


OPTIMIZADORES ={
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RMSprop": torch.optim.RMSprop,
    "LM": LevenberMaquardtOpt
}

FN_LOSS ={
    "MSE": torch.nn.MSELoss,
    "MAE": torch.nn.L1Loss,
    "SSE": lambda output, target: torch.sum((output - target) ** 2),
    "Entropia": torch.nn.CrossEntropyLoss
}

def cargar_datos(path, data_config:dict)->pd.DataFrame:
    #si los datos estan separados en input, target
    if type(path) == list: #siempre 2 (in y out)
        if path[0][-3:] == "mat":
            inputs_str = data_config["header"]["in"]
            outputs_str = data_config["header"]["out"]
            din = loadmat(path[0])[inputs_str]
            dout = loadmat(path[1])[outputs_str]
            #asumiendo que esta como caracteristicas renglones y muestras columnas
            #tipo matlab
            data = np.vstack([din,dout])
            return pd.DataFrame(data.T)
    elif type(path) == str:
        if path[-3:] == "mat":
            data_dict = loadmat(path)
            return pd.DataFrame(data_dict)
        elif path[-4:] == "data" or path[-3:] == "csv":
            return pd.read_csv(path,**data_config)
    return None

def main(config:DataConfig):
    
    # cargar datos
    dataset_file = config.experimentos.dataset_path
    resultados_path = config.experimentos.resultados_path
    exp_corridas = config.experimentos.corridas
    
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_conf = {
        "header":config.experimentos.dataset_header,
    }
    
    #crear folders
    os.makedirs(resultados_path,exist_ok=True)
    
    #cargar datos
    
    datos = cargar_datos(dataset_file, data_conf)
    
    if config.experimentos.dataset_target_col !=None:
        data_in = datos.drop(columns=[config.experimentos.dataset_target_col]).to_numpy()
    else:
        data_in = datos[datos.columns[:config.experimentos.dataset_entradas]].to_numpy()
        
    if config.experimentos.dataset_map_col !=None:
        data_out = datos[config.experimentos.dataset_target_col].map(config.experimentos.dataset_map_col).to_numpy()
    else:
        data_out = datos[datos.columns[
            config.experimentos.dataset_entradas:
            config.experimentos.dataset_entradas+config.experimentos.dataset_salidas
            ]].to_numpy()
    
    
    estado={
        "iter_act":config.experimentos.reglas_inicial,
        "iter_total":config.experimentos.reglas_total,
        "flair": 'reglas evaluadas',
        "color": "\033[1;44;33m",
        "cursor":"\033[1;0H"
    }
    estadisticas={}
    for nombres in OPTIMIZADORES:
        estadisticas[nombres]={}
        
    stop_event = threading.Event()
    barra_reglas = threading.Thread(target=mostrar_barra_progreso,args=(estado,stop_event),daemon=True)
    barra_reglas.start()
    semilla={}
    #semilla igual para todos los optimizadores, 
    for i in range(config.experimentos.reglas_inicial,config.experimentos.reglas_total+1):
        semilla[i] = np.random.randint(0,9999)
        
    for regla in range(
        config.experimentos.reglas_inicial,
        config.experimentos.reglas_total+1):
        
        estado["iter_act"] = regla
        
        
        # generar datos dummy para el fis
        d_train_x, d_x_temp, d_train_y, d_y_temp = train_test_split(data_in,data_out,test_size=config.experimentos.test_size)
        
    
        # generar los modelos y convertir a tensores
    
        train_y = torch.from_numpy(d_train_y).to(torch.float64)

    
        # onehotencode si es clasificación

        if config.experimentos.tipo == "clasificacion":
            train_y = OneHotEncode(train_y,config.experimentos.dataset_salidas)
            #test_y = OneHotEncode(test_y,config.experimentos.dataset_salidas)
            #val_y = OneHotEncode(val_y,config.experimentos.dataset_salidas)

        nombre = config.experimentos.dataset_path.split("/")[-1] if type(config.experimentos.dataset_path) !=list \
        else config.experimentos.dataset_path[0].split("/")[-1]

        fis, fis_str = CrearFISInicial(nombre+"_inicial.fis",
                                    pd.DataFrame(d_train_x),
                                    pd.DataFrame(train_y.numpy().squeeze()),
                                    regla)
    
        for m in config.optimizadores:
            #print(f"\033[1;45;33m\nEvaluando modelo con optimizador-> {m.nombre} y regla {regla}\033[0m")
            
            m_nom = m.nombre
            estadisticas[m.nombre][regla]={
                "losses":[],
                "prom_loss":0,
                "prom_epocas":0,
                "r2s":[],
                "prom_r2":0,
                "accuracys":[],
                "prom_acc":0,
                "presicions":[],
                "prom_prec":0
            }
            
            if not m_nom in OPTIMIZADORES:
                raise ValueError(f"Optimizador '{m_nom}' no está config")
            
            opt = OPTIMIZADORES[m_nom]
            if m_nom == "LM":
                m.params["device"]=gpu_device
            
            if not config.experimentos.funcion_perdida in FN_LOSS:
                raise ValueError(f"Función de perdida '{config.experimentos.funcion_perdida}' no está config")
            
            loss_fn = FN_LOSS[config.experimentos.funcion_perdida]
            
            # hacer n corridas y calcular el promedio por modelo
            for corrida in range(exp_corridas):
                modelo = RLANFISBuilder() \
                        .AddFIS(fis_str)\
                        .AddInputs(config.experimentos.dataset_entradas)\
                        .AddOutputs(config.experimentos.dataset_salidas)\
                        .AddRules(regla)\
                        .AddValMaxFails(config.experimentos.fallos)\
                        .AddOptimizador(opt,**m.params)\
                        .AddFunctionLoss(loss_fn)\
                        .Build()
                #modelos[m_nom]= modelo
                
                modelo = modelo.to(gpu_device)

                estado["flair"] = f"evaluando reglas | {m.nombre} | corrida: {corrida+1}/{exp_corridas}"
            
                # separar los datos entrenamiento, prueba y validacion
                semilla_corrida = semilla[regla]+corrida #misma semilla para cada optimizador
                d_train_x, d_x_temp, d_train_y, d_y_temp = train_test_split(data_in,data_out,test_size=config.experimentos.test_size,random_state=semilla_corrida)
                d_test_x, d_val_x, d_test_y, d_val_y = train_test_split(d_x_temp,d_y_temp,test_size=config.experimentos.val_size,random_state=semilla_corrida)
            
                # generar los modelos y convertir a tensores
            
                train_x = torch.from_numpy(d_train_x).to(torch.float64).to(gpu_device)
                train_y = torch.from_numpy(d_train_y).to(torch.float64).to(gpu_device)

                test_x = torch.from_numpy(d_test_x).to(torch.float64).to(gpu_device)
                test_y = torch.from_numpy(d_test_y).to(torch.float64).to(gpu_device)

                val_x = torch.from_numpy(d_val_x).to(torch.float64).to(gpu_device)
                val_y = torch.from_numpy(d_val_y).to(torch.float64).to(gpu_device)
                
                if config.experimentos.tipo == "clasificacion":
                    train_y = OneHotEncode(train_y,config.experimentos.dataset_salidas).to(gpu_device)
                    test_y = OneHotEncode(test_y,config.experimentos.dataset_salidas).to(gpu_device)
                    val_y = OneHotEncode(val_y,config.experimentos.dataset_salidas).to(gpu_device)
            
                
                
                hist_loss = train_nfs(modelo,train_x,train_y,
                                    config.experimentos.epocas,config.experimentos.tolerancia,
                                    debug=False)
                
                y_test = modelo(test_x)
                if config.experimentos.tipo == "clasificacion":
                    #calcular las precicion y la exactitud
                    acc,prec = confusion_matrix(y_test,test_y,num_classes=config.experimentos.dataset_salidas,plot=False)
                    estadisticas[m.nombre][regla]["accuracys"].append(acc)
                    estadisticas[m.nombre][regla]["presicions"].append(prec)
                else:
                    #calcular la r2
                    r,r2= PlotTraining(test_x,y_test,test_y,plot=False,debug=False)
                    estadisticas[m.nombre][regla]["r2s"].append(r2)
                    pass
                
                estadisticas[m.nombre][regla]["losses"] += hist_loss
            estadisticas[m.nombre][regla]["prom_loss"]= np.average(estadisticas[m.nombre][regla]["losses"])
            estadisticas[m.nombre][regla]["prom_epocas"] = len(estadisticas[m.nombre][regla]["losses"])//exp_corridas
            if config.experimentos.tipo == "clasificacion":
                estadisticas[m.nombre][regla]["prom_prec"] = np.average(estadisticas[m.nombre][regla]["presicions"])
                estadisticas[m.nombre][regla]["prom_acc"]= np.average(estadisticas[m.nombre][regla]["accuracys"])
            else:
                estadisticas[m.nombre][regla]["prom_r2"]= np.average(estadisticas[m.nombre][regla]["r2s"])
                
            sys.stdout.write("\033[2K") #borrar linea
            sys.stdout.write("\033[2A") #dos arriba renglon
            sys.stdout.write("\033[2K") # borrar linea
                
    stop_event.set()
    barra_reglas.join()
            
    # generar graficas y estadisticas
    guardar_resultados(config.experimentos,estadisticas)
        
    # guardar graficas de los resultados
    print(f"resultados guardados en -> {resultados_path}resultados.csv")
    
    return

def guardar_resultados(experimento:DataExperimento, resultados:dict)->None:
    
    with open(experimento.resultados_path+"info_experimentos.txt","w+") as txt:
        txt.write(f"Los experimentos para {experimento.dataset_path} se realizaron de la forma: \n")
        txt.write(f"\t- Se Entreno y probo {experimento.corridas} veces, el resultado se promedio y eso es lo que se reporta\n")
        txt.write(f"\t- Los datos se dividieron en entrenamiento, prueba y validación aleatorizados con la misma semilla por optimizador \n")
        txt.write(f"\t- Se tomaron en cuenta las siguientes metricas: \n")
        txt.write(f"\t\t* Para clasificación la exactitud(accuracy) y presición(precision) \n")
        txt.write(f"\t\t* Para regresión la R^2 \n")
        txt.write(f"\t\t* Para ambos, la cantidad de epocas promedio máximo de {experimento.epocas}, el valor promedio de la función de perdida \n")
    
    with open(experimento.resultados_path+"resultados.csv","w+") as csv:
        header ="optimizador,metrica"
        for i in range(experimento.reglas_inicial,experimento.reglas_total+1):
            header+= f",regla_{i}"
        csv.write(header+"\n")
        for modelo in resultados:
            renglon=modelo
            regla_loss=",loss"
            regla_epocas=",epocas"
            regla_acc = ",exactitud"
            regla_prec = ",presición"
            regla_r2 = ",R2"
            for regla in range(experimento.reglas_inicial,experimento.reglas_total+1):
                regla_loss += f",{resultados[modelo][regla]["prom_loss"]}"
                regla_epocas += f",{resultados[modelo][regla]["prom_epocas"]}"
                regla_acc += f",{resultados[modelo][regla]["prom_acc"]}"
                regla_prec += f",{resultados[modelo][regla]["prom_prec"]}"
                regla_r2 += f",{resultados[modelo][regla]["prom_r2"]}"
            csv.write(renglon+regla_loss+"\n")
            csv.write(renglon+regla_epocas+"\n")
            if experimento.tipo == "clasificacion":
                csv.write(renglon+regla_acc+"\n")
                csv.write(renglon+regla_prec+"\n")
            else:
                csv.write(renglon+regla_r2+"\n")



if __name__ == "__main__":
    help = """
        Args:
        [0] - script name
        [1] - json con la configuración de los experimentos (e.g. "iris_config.json")
        [2] - usar dispositivo (cuda, mps, sin nada es cpu)         
        Json config ejemplo:
        {
            "optimizadores": 
            [
                {
                    "name": "LM",
                    "params": {
                        "lambda_init":0.01,
                        "lambda_decr":0.9,
                        "lambda_incr":10
                    }
                },
                {
                    "name": "SGD",
                    "params": {
                        "learning_rate": 0.01,
                        "momentum": 0.9
                    }
                },
                {
                    "name": "Adam",
                    "params": {
                        "learning_rate": 0.01,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "epsilon": 1e-8
                    }
                }
            ],
            "experimentacion":
            [
                {
                    "dataset_path": "dataset/iris.mat",
                    "resultados_path": "resultados/iris/",
                    "corridas": 100,
                    "epocas": 1000,
                    "tolerancia": 1e-12,
                    "fallos" : 20
                }
            ]        
        }
    """
    
    if len(sys.argv) < 2:
        print(help)
        exit(1)
    
    if len(sys.argv) == 3:
        gpu_device = sys.argv[2]
        
    json_path = sys.argv[1]
    with open(json_path,'r') as file:
        config = json.load(file)
    
    data_config = parsear_json(config)
    
    timer_start()
    main(data_config)
    timer_end()
    print(f"se ejecuto el script en\n{formato_legible(timer)}")
        