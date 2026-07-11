import sys
import numpy as np
import json
from funciones_auxiliares import DataOptimizador,DataExperimento,DataConfig,OneHotEncode
from rpipeline import parsear_json, cargar_datos,fis_nan_values_edit,timer_start, timer_end, formato_legible
import pandas as pd
from sklearn.model_selection import train_test_split
from V2_Anfis import LevenberMaquardtOpt,RLANFISBuilder,train_nfs
from Anfis_utils import CrearFISInicial
import torch

OPTIMIZADORES ={
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RMSprop": torch.optim.RMSprop,
    "AdamFactor": torch.optim.Adafactor,
    "LM": LevenberMaquardtOpt
}


def calcular_muestra_sig(n: int, p: float, z: float, e: float) -> int:
    """
    Calcula el tamaño de muestra necesario para una proporción poblacional.

    Parámetros:
    n (int): Tamaño de la población.\n
    p (float): Proporción estimada de la población (entre 0 y 1).\n
    z (float): Valor crítico de z para el nivel de confianza deseado.\n
    e (float): Margen de error permitido (entre 0 y 1).\n

    Retorna:
    int: Tamaño de muestra necesario.
    """
    
    numerator = (z ** 2) * p * (1 - p)
    denominator = e ** 2
    sample_size = numerator / denominator

    adjusted_sample_size = sample_size / (1 + ((sample_size - 1) / n))

    return np.ceil(adjusted_sample_size).astype(np.int16)


def encontrar_parametros(config: DataConfig) ->None:
    global path_archivo
    global exp_nombre
    dataset_file = config.experimentos.dataset_path
    #resultados_path = config.experimentos.resultados_path
    #exp_corridas = config.experimentos.corridas
    
    data_conf = {
        "header":config.experimentos.dataset_header,
    }
    if config.experimentos.dataset_sep !=None:
        data_conf["sep"]=config.experimentos.dataset_sep
    
    datos = cargar_datos(dataset_file, data_conf)
    
    muestra_sig= calcular_muestra_sig(datos.shape[0], 0.5, 1.96, 0.05)
    
    print(f"muestras de datos: {datos.shape[0]}")
    print(f"muestra significativa: {muestra_sig}")
    
    #cargar datos
    if config.experimentos.dataset_target_col !=None:
        data_in = datos.drop(columns=datos.columns[config.experimentos.dataset_target_col]).to_numpy()
    else:
        data_in = datos[datos.columns[:config.experimentos.dataset_entradas]].to_numpy()
        
    if config.experimentos.dataset_map_col !=None:
        data_out = datos[datos.columns[config.experimentos.dataset_target_col]].map(config.experimentos.dataset_map_col).to_numpy()
    else:
        data_out = datos[datos.columns[
            config.experimentos.dataset_entradas:
            config.experimentos.dataset_entradas+config.experimentos.dataset_salidas
            ]].to_numpy()
    
    d_train_x, d_x_test, d_train_y, d_y_test = train_test_split(data_in,data_out,test_size=muestra_sig)
    
    d_train_x = torch.from_numpy(d_train_x).to(torch.float64)
    d_x_test = torch.from_numpy(d_x_test).to(torch.float64)
    d_train_y = torch.from_numpy(d_train_y).to(torch.float64)
    d_y_test = torch.from_numpy(d_y_test).to(torch.float64)
    
    if config.experimentos.tipo == "clasificacion":
            d_train_y = OneHotEncode(d_train_y,config.experimentos.dataset_salidas)
            d_y_test = OneHotEncode(d_y_test,config.experimentos.dataset_salidas)
    
    lr_min = 1e-12
    lr_max = 0.99
    parametros={}
    for optimizador in config.optimizadores:
        parametros[optimizador.nombre]={}
    
    nombre_data = config.experimentos.dataset_path.split("/")[-1] if type(config.experimentos.dataset_path) !=list \
        else config.experimentos.dataset_path[0].split("/")[-1]
        
    loss_fn = lambda output, target: torch.sum((output - target) ** 2)
    
    print("\033[2J\033[1;1H",end="")

    for regla in range(config.experimentos.reglas_inicial,config.experimentos.reglas_total+1):
        
        fis, fis_str = CrearFISInicial("mparams_"+nombre_data+"_inicial.fis",
                                    pd.DataFrame(d_train_x.numpy()),
                                    pd.DataFrame(d_train_y.numpy().squeeze()),
                                    regla)
        fis_nan_values_edit(fis_str)
        
        for optimizador in config.optimizadores:
            best_lr =1e-3 #por defecto
            best_loss = float('inf')
            parametros[optimizador.nombre][regla]=optimizador.params
            print(f"Regla: {regla}/{config.experimentos.reglas_total}, {optimizador.nombre}")
            for lr in np.logspace(np.log10(lr_max), np.log10(lr_min), num=100):
                
                if optimizador.nombre !="LM":
                    optimizador.params["lr"] = lr 
                else:
                    optimizador.params["lambda_init"] = lr 
                
                modelo = RLANFISBuilder() \
                        .AddFIS(fis_str)\
                        .AddInputs(config.experimentos.dataset_entradas)\
                        .AddOutputs(config.experimentos.dataset_salidas)\
                        .AddRules(regla)\
                        .AddValMaxFails(config.experimentos.fallos)\
                        .AddOptimizador(OPTIMIZADORES[optimizador.nombre],**optimizador.params)\
                        .AddFunctionLoss(loss_fn)\
                        .Build()
                
                hist_loss, _ = train_nfs(modelo,d_train_x,d_train_y,
                                    config.experimentos.epocas,config.experimentos.tolerancia,
                                    debug=False)
                
                promedio_loss = np.mean(hist_loss)
                
                if promedio_loss < best_loss and (np.isnan(hist_loss[-1]) == False or np.isinf(hist_loss[-1]) == False):
                    best_loss = promedio_loss
                    best_lr = lr
            
            if optimizador.nombre != "LM":
                parametros[optimizador.nombre][regla]["lr"]=best_lr
            else:
                parametros[optimizador.nombre][regla]["lambda_init"]=best_lr
            print("\033[2J\033[1;1H")
        
    pjson = json.dumps(parametros, indent=4)
    if path_archivo == "":
        path_archivo = "./"
    json.dump(parametros,open(path_archivo+f"mejores_parametros_optimizadores_{exp_nombre}.json","w+"),indent=4)


path_archivo :str=""
exp_nombre :str=""
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usar: python mejores_parametros.py dataset_json.json")
        sys.exit(1)

    path_archivo = sys.argv[1]
    with open(path_archivo,'r') as file:
        config = json.load(file)
    
    exp_nombre = sys.argv[1].split("/")[-1]
    path_archivo = path_archivo.split(".")[0]
    temp = path_archivo.split("/")[-1]
    path_archivo = path_archivo.replace(temp,"")
    
    data_config = parsear_json(config)
    
    timer_start()
    encontrar_parametros(data_config)
    timer = timer_end()
    print(f"Tiempo total de ejecución: {formato_legible(timer)}")
    


