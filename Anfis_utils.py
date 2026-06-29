#archivo con funciones de ayuda
import numpy as np
import pandas as pd
import fuzzylab as fz
from scipy.io import loadmat


#
def CrearFISInicial(nombre:str,datos_in:pd.DataFrame,datos_out:pd.DataFrame, num_reglas:int)->tuple[fz.mamfis,str]:
    
    fis = fz.mamfis(nombre+"_init")
    num_entradas = datos_in.shape[1]
    num_salidas = datos_out.shape[1]
    mf_n = np.ones(num_entradas)*num_reglas
    epsilon = 1e-8
    rango = np.vstack((np.min(datos_in.to_numpy(),axis=0)-epsilon  ,np.max(datos_in.to_numpy(),axis=0)+epsilon )).T
    rango_out = np.vstack((np.min(datos_out.to_numpy(),axis=0),np.max(datos_out.to_numpy(),axis=0))).T
    for e in range(num_entradas):
        dmin ,dmax = rango[e]
        fis.addInput([float(dmin),float(dmax)],Name="Entrada"+str(e))
        a = (rango[e,1] - rango[e, 0])/2/(mf_n[e] - 1);
        a = a/np.sqrt(2*np.log(2));
        c = np.linspace(rango[e, 0], rango[e, 1], int(mf_n[e]));
        for mf in range(num_reglas):
            fis.addMF("Entrada"+str(e),"gaussmf",[float(a),float(c[mf])],Name="MF"+str(mf))
    #output es constante por lo que en 0 esta bien las mf
    for s in range(num_salidas):
        domin ,domax = rango_out[s]
        fis.addOutput([float(domin),float(domax)],Name="Salida"+str(s))
        for mf in range(num_reglas):
            fis.addMF("Salida"+str(s),"constant",[0],Name="MF"+str(mf))
    
    #reglas
    reglas = []
    for r in range(num_reglas):
        ri = [r+1]*num_entradas + [r+1]*num_salidas + [1]*2
        reglas.append(ri)
    fis.addRule(reglas)
    fz.writeFIS(fis,nombre+"__init.fis")
    
    return fis, nombre+"__init.fis"


def CargarFIS(nombre:str, )->tuple[fz.fisvar,np.array,np.array]:
    fis = fz.readfis(nombre)

    num_in = len(fis.Inputs)
    num_out = len(fis.Outputs)
    num_rules = len(fis.Rules)

    #print(f"Cargando FIS\nEntradas: {num_in} Salidas: {num_out} Reglas:{num_rules}")

    #por el momento solo para guasiana (sigma, centro)
    m_in = np.zeros((num_in, num_rules,2))
    # por el momento solo con constante
    m_out = np.zeros((num_out, num_rules,1))

    for i in range(num_in):
        for r in range(num_rules):
            id = fis.Rules[r].Antecedent[i]
            m_in[i,r,0]= fis.Inputs[i].MembershipFunctions[id-1].Parameters[0]
            m_in[i,r,1]= fis.Inputs[i].MembershipFunctions[id-1].Parameters[1]

    for o in range(num_out):
        for r in range(num_rules):
            id = fis.Rules[r].Consequent[o]
            m_out[o,r,0]= fis.Outputs[o].MembershipFunctions[id-1].Parameters[0]

    return (fis, m_in, m_out) 

def GuardarFIS(nombre:str, fis:fz.fisvar,modelo)->None:
    reglas = fis.Rules
    num_in = len(fis.Inputs)
    parametros = modelo.parameters() # iterador de cada unos de los parametros 
    #for param in parametros:
    centro = parametros.__next__()
    sigma  = parametros.__next__()
    theta  = parametros.__next__()
    for i in range(num_in):
        for r in range(len(reglas)):
            id = reglas[r].Antecedent[i]
            fis.Inputs[i].MembershipFunctions[id-1].Parameters[1]=float(centro[i,r].detach().numpy())
            fis.Inputs[i].MembershipFunctions[id-1].Parameters[0]=float(sigma[i,r].detach().numpy())
    
    for rr in range(len(reglas)):
        num_cos = len(reglas[rr].Consequent)
        for o in range(num_cos):
            oid = reglas[rr].Consequent[o]
            fis.Outputs[o].MembershipFunctions[oid-1].Parameters[0] = float(theta.data[r,o].detach().numpy())
            
    fz.writeFIS(fis,nombre)