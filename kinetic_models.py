
import parameter_function as par
import networkx as nx   #to deal with network
import numpy as np
import sympy as sym
from sympy import *
from sympy.printing.mathml import mathml
init_printing(use_unicode=True) # allow LaTeX printing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random 
from collections import Counter

# =============================================================================
# STEADY-STATE COMPUTATION for different models
# =============================================================================

'''
Possible Function to use for the drug modeling:

    ### Decreasing Functions

- Hyperbolic Inibition: k1 / (1 + (Drug/Kd))
- Hill function: k1 * (1- (Drug**n / (k**n  + Drug**n)))
- Exponential Decay: k1 * exp(-lambda * Drug)

    ### Increasing Functions
- Linear Modulation: k2 ( 1 + beta * Drug )
- Hill activation = k2 + (Vmax - k2) * (Drug**n / (hill_cost**n  + Drug**n))

'''

# =============================================================================
# Model 1
# =============================================================================

def BP_linear_dissociation(
    drug_conc: list[float],
    epochs:    int,
) -> list[dict]:
    """
    Calcola le probabilità allo steady state per un modello a 3 nodi
    su una griglia di concentrazioni di farmaco e set di parametri sintetici.

    Topologia del grafo (DiGraph pesato, 3 nodi):
        1 (libero) <──> 2 (sito corretto)
        1 (libero) <──> 3 (sito errato / decoy)

    Effetto lineare del farmaco sulla dissociazione dal sito corretto :
        k_1_drug = k_1_basal + alpha * drug

    Output (lista per set di parametri):
        {
            'params':  { k1, k_1_basal, k2, k_2, alpha, n },
            'results': [ { 'drug': ..., 'P1': ..., 'P2': ..., 'P3': ... }, ... ]
        }
    """
    parameters_sets = par.par.generate_parameter_sets(epochs)
    all_results = []

    for param_set in parameters_sets:
        k1        = float(param_set['k1'])
        k_1 =       float(param_set['k_1'])
        k2        = float(param_set['k2'])
        k_2       = float(param_set['k_2'])
        alpha     = float(param_set['alpha'])
        n         = int(param_set['n'])

        drug_results = []

        for drug in drug_conc:
            # --- Effetto lineare del farmaco sulla dissociazione corretta ---

            k_1_drug  = k_1 + alpha * drug

            # --- Costruzione grafo diretto ---
            G = nx.DiGraph()
            G.add_nodes_from([1, 2, 3])
            G.add_weighted_edges_from([
                (1, 2, k1),       # associazione al sito corretto
                (2, 1, k_1_drug), # dissociazione dal sito corretto (drug-modulated)
                (1, 3, k2),       # associazione al sito errato
                (3, 1, k_2),      # dissociazione dal sito errato
            ])

            # --- Matrix-Tree Theorem su grafo invertito ---
            G_rev = G.reverse()
            rho = {
                node: nx.number_of_spanning_trees(G_rev, root=node, weight="weight")
                for node in [1, 2, 3]
            }

            Z = sum(rho.values())
            if Z == 0:
                raise ValueError(
                    f"Z = 0 per params={param_set}, drug={drug}. "
                    "Verificare connettività e pesi del grafo."
                )

            drug_results.append({
                'drug':       drug,
                'hill_effect': 0,
                'k_1_drug':   round(k_1_drug, 6),
                'P1':         rho[1] / Z,
                'P2':         rho[2] / Z,
                'P3':         rho[3] / Z,
            })

        all_results.append({
            'params':  param_set,
            'results': drug_results,
        })

    return all_results

# =============================================================================
# Model 2
# =============================================================================



def BP_hill_dissociation(
    drug_conc: list[float],
    epochs:    int,
    hill_costant: float
) -> list[dict]:
    """
    Calcola le probabilità allo steady state per un modello a 3 nodi
    su una griglia di concentrazioni di farmaco e set di parametri sintetici.

    Topologia del grafo (DiGraph pesato, 3 nodi):
        1 (libero) <──> 2 (sito corretto)
        1 (libero) <──> 3 (sito errato / decoy)

    Effetto del farmaco sulla dissociazione dal sito corretto :
        k_1_drug = k_1_basal + alpha * drug^n / (K_hill^n + drug^n)

    Output (lista per set di parametri):
        {
            'params':  { k1, k_1_basal, k2, k_2, alpha, n },
            'results': [ { 'drug': ..., 'P1': ..., 'P2': ..., 'P3': ... }, ... ]
        }
    """
    parameters_sets = par.par.generate_parameter_sets(epochs)
    all_results = []

    for param_set in parameters_sets:
        k1        = float(param_set['k1'])
        k_1_basal = float(param_set['k_1'])
        k2        = float(param_set['k2'])
        k_2       = float(param_set['k_2'])
        alpha     = float(param_set['alpha'])
        n         = int(param_set['n'])

        drug_results = []

        for drug in drug_conc: 

            # --- Effetto Hill del farmaco sulla dissociazione corretta ---

            hill_fun  = (drug ** n) / (hill_costant**n + drug ** n) if drug > 0 else 0.0
            k_1_drug  = k_1_basal + hill_fun

            # --- Costruzione grafo diretto ---
            G = nx.DiGraph()
            G.add_nodes_from([1, 2, 3])
            G.add_weighted_edges_from([
                (1, 2, k1),       # associazione al sito corretto
                (2, 1, k_1_drug), # dissociazione dal sito corretto (drug-modulated)
                (1, 3, k2),       # associazione al sito errato
                (3, 1, k_2),      # dissociazione dal sito errato
            ])

            # --- Matrix-Tree Theorem su grafo invertito ---
            G_rev = G.reverse()
            rho = {
                node: nx.number_of_spanning_trees(G_rev, root=node, weight="weight")
                for node in [1, 2, 3]
            }

            Z = sum(rho.values())
            if Z == 0:
                raise ValueError(
                    f"Z = 0 per params={param_set}, drug={drug}. "
                    "Verificare connettività e pesi del grafo."
                )

            drug_results.append({
                'drug':       drug,
                'hill_effect': round(hill_fun, 6),
                'k_1_drug':   round(k_1_drug, 6),
                'P1':         rho[1] / Z,
                'P2':         rho[2] / Z,
                'P3':         rho[3] / Z,
            })

        all_results.append({
            'params':  param_set,
            'results': drug_results,
        })

    return all_results


# =============================================================================
# Model 3
# =============================================================================

def Decoy_stabilization(
    drug_conc: list[float],
    epochs:    int,
    hill_costant: float
) -> list[dict]:
    """
    Calcola le probabilità allo steady state per un modello a 3 nodi
    su una griglia di concentrazioni di farmaco e set di parametri sintetici.

    Topologia del grafo (DiGraph pesato, 3 nodi):
        1 (libero) <──> 2 (sito corretto)
        1 (libero) <──> 3 (sito errato / decoy)

    Effetto del farmaco sulla dissociazione dal sito corretto :
        k_1_drug = k_1 + alpha * drug
        k2_drug = k2 + drug^n / (K_hill^n + drug^n)

    Output (lista per set di parametri):
        {
            'params':  { k1, k_1_basal, k2, k_2, alpha, n },
            'results': [ { 'drug': ..., 'P1': ..., 'P2': ..., 'P3': ... }, ... ]
        }
    """
    parameters_sets = par.generate_parameter_sets(epochs)
    all_results = []

    for param_set in parameters_sets:
        k1        = float(param_set['k1'])
        k_1_basal = float(param_set['k_1'])
        k2        = float(param_set['k2'])
        k_2       = float(param_set['k_2'])
        alpha     = float(param_set['alpha'])
        n         = int(param_set['n'])

        drug_results = []

        for drug in drug_conc: 


            
            k_2_drug  = k_2 + (drug/hill_costant)

            # --- Costruzione grafo diretto ---
            G = nx.DiGraph()
            G.add_nodes_from([1, 2, 3])
            G.add_weighted_edges_from([
                (1, 2, k1),       # associazione al sito corretto
                (2, 1, k_1_basal), # dissociazione dal sito corretto (drug-modulated)
                (1, 3, k2),  # associazione al sito errato (drug-modulated)
                (3, 1, k_2_drug),      # dissociazione dal sito errato
            ])

            # --- Matrix-Tree Theorem su grafo invertito ---
            G_rev = G.reverse()
            rho = {
                node: nx.number_of_spanning_trees(G_rev, root=node, weight="weight")
                for node in [1, 2, 3]
            }

            Z = sum(rho.values())
            if Z == 0:
                raise ValueError(
                    f"Z = 0 per params={param_set}, drug={drug}. "
                    "Verificare connettività e pesi del grafo."
                )

            drug_results.append({
                'drug':       drug,
                'k_2_drug':   round(k_2_drug, 6),
                'P1':         rho[1] / Z,
                'P2':         rho[2] / Z,
                'P3':         rho[3] / Z,
            })

        all_results.append({
            'params':  param_set,
            'results': drug_results,
        })

    return all_results


# =============================================================================
# Model 4
# =============================================================================

def Decoy_assotiation_hill(
    drug_conc: list[float],
    epochs:    int,
    hill_costant: float
) -> list[dict]:
    """
    Calcola le probabilità allo steady state per un modello a 3 nodi
    su una griglia di concentrazioni di farmaco e set di parametri sintetici.

    Topologia del grafo (DiGraph pesato, 3 nodi):
        1 (libero) <──> 2 (sito corretto)
        1 (libero) <──> 3 (sito errato / decoy)

    Effetto del farmaco sulla dissociazione dal sito corretto :
        k_1_drug = k_1 + alpha * drug
        k2_drug = k2 + drug^n / (K_hill^n + drug^n)

    Output (lista per set di parametri):
        {
            'params':  { k1, k_1_basal, k2, k_2, alpha, n },
            'results': [ { 'drug': ..., 'P1': ..., 'P2': ..., 'P3': ... }, ... ]
        }
    """
    parameters_sets = par.generate_parameter_sets(epochs)
    all_results = []

    for param_set in parameters_sets:
        k1        = float(param_set['k1'])
        k_1_basal = float(param_set['k_1'])
        k2        = float(param_set['k2'])
        k_2       = float(param_set['k_2'])
        alpha     = float(param_set['alpha'])
        n         = int(param_set['n'])

        drug_results = []

        for drug in drug_conc: 

            # --- Effetto Hill del farmaco sulla associazione scorretta ---

            hill_fun  = (drug ** n) / (hill_costant**n + drug ** n) if drug > 0 else 0.0
            k2_drug  = k2 + hill_fun

            # --- Costruzione grafo diretto ---
            G = nx.DiGraph()
            G.add_nodes_from([1, 2, 3])
            G.add_weighted_edges_from([
                (1, 2, k1),       # associazione al sito corretto
                (2, 1, k_1_basal), # dissociazione dal sito corretto (drug-modulated)
                (1, 3, k2_drug),  # associazione al sito errato (drug-modulated)
                (3, 1, k_2),      # dissociazione dal sito errato
            ])

            # --- Matrix-Tree Theorem su grafo invertito ---
            G_rev = G.reverse()
            rho = {
                node: nx.number_of_spanning_trees(G_rev, root=node, weight="weight")
                for node in [1, 2, 3]
            }

            Z = sum(rho.values())
            if Z == 0:
                raise ValueError(
                    f"Z = 0 per params={param_set}, drug={drug}. "
                    "Verificare connettività e pesi del grafo."
                )

            drug_results.append({
                'drug':       drug,
                'hill_effect': round(hill_fun, 6),
                'k2_drug':   round(k2_drug, 6),
                'P1':         rho[1] / Z,
                'P2':         rho[2] / Z,
                'P3':         rho[3] / Z,
            })

        all_results.append({
            'params':  param_set,
            'results': drug_results,
        })

    return all_results
# =============================================================================
# Model 5
# =============================================================================
def BP_linear_Decoy_assotiation_hill(
    drug_conc: list[float],
    epochs:    int,
    hill_costant: float
) -> list[dict]:
    """
    Calcola le probabilità allo steady state per un modello a 3 nodi
    su una griglia di concentrazioni di farmaco e set di parametri sintetici.

    Topologia del grafo (DiGraph pesato, 3 nodi):
        1 (libero) <──> 2 (sito corretto)
        1 (libero) <──> 3 (sito errato / decoy)

    Effetto del farmaco sulla dissociazione dal sito corretto :
        k_1_drug = k_1 + alpha * drug
        k2_drug = k2 + drug^n / (K_hill^n + drug^n)

    Output (lista per set di parametri):
        {
            'params':  { k1, k_1_basal, k2, k_2, alpha, n },
            'results': [ { 'drug': ..., 'P1': ..., 'P2': ..., 'P3': ... }, ... ]
        }
    """
    parameters_sets = par.generate_parameter_sets(epochs)
    all_results = []

    for param_set in parameters_sets:
        k1        = float(param_set['k1'])
        k_1_basal = float(param_set['k_1'])
        k2        = float(param_set['k2'])
        k_2       = float(param_set['k_2'])
        alpha     = float(param_set['alpha'])
        n         = int(param_set['n'])

        drug_results = []

        for drug in drug_conc: 
            # --- Effetto lineare del farmaco sulla dissociazione corretta ---
            k_1_drug  = k_1_basal + alpha * drug

            # --- Effetto Hill del farmaco sulla associazione scorretta ---

            hill_fun  = (drug ** n) / (hill_costant**n + drug ** n) if drug > 0 else 0.0
            k2_drug  = k2 + hill_fun

            # --- Costruzione grafo diretto ---
            G = nx.DiGraph()
            G.add_nodes_from([1, 2, 3])
            G.add_weighted_edges_from([
                (1, 2, k1),       # associazione al sito corretto
                (2, 1, k_1_drug), # dissociazione dal sito corretto (drug-modulated)
                (1, 3, k2_drug),  # associazione al sito errato (drug-modulated)
                (3, 1, k_2),      # dissociazione dal sito errato
            ])

            # --- Matrix-Tree Theorem su grafo invertito ---
            G_rev = G.reverse()
            rho = {
                node: nx.number_of_spanning_trees(G_rev, root=node, weight="weight")
                for node in [1, 2, 3]
            }

            Z = sum(rho.values())
            if Z == 0:
                raise ValueError(
                    f"Z = 0 per params={param_set}, drug={drug}. "
                    "Verificare connettività e pesi del grafo."
                )

            drug_results.append({
                'drug':       drug,
                'hill_effect': round(hill_fun, 6),
                'k_1_drug':   round(k_1_drug, 6),
                'k2_drug':   round(k2_drug, 6),
                'P1':         rho[1] / Z,
                'P2':         rho[2] / Z,
                'P3':         rho[3] / Z,
            })

        all_results.append({
            'params':  param_set,
            'results': drug_results,
        })

    return all_results

# =============================================================================
# Model 6
# =============================================================================
def BP_linear_Decoy_stabilization(
    drug_conc: list[float],
    epochs:    int,
    hill_costant: float
) -> list[dict]:
    """
    Calcola le probabilità allo steady state per un modello a 3 nodi
    su una griglia di concentrazioni di farmaco e set di parametri sintetici.

    Topologia del grafo (DiGraph pesato, 3 nodi):
        1 (libero) <──> 2 (sito corretto)
        1 (libero) <──> 3 (sito errato / decoy)

    Effetto del farmaco sulla dissociazione dal sito corretto :
        k_1_drug = k_1 + alpha * drug
        k2_drug = k2 + drug^n / (K_hill^n + drug^n)

    Output (lista per set di parametri):
        {
            'params':  { k1, k_1_basal, k2, k_2, alpha, n },
            'results': [ { 'drug': ..., 'P1': ..., 'P2': ..., 'P3': ... }, ... ]
        }
    """
    parameters_sets = par.generate_parameter_sets(epochs)
    all_results = []

    for param_set in parameters_sets:
        k1        = float(param_set['k1'])
        k_1_basal = float(param_set['k_1'])
        k2        = float(param_set['k2'])
        k_2       = float(param_set['k_2'])
        alpha     = float(param_set['alpha'])
        n         = int(param_set['n'])

        drug_results = []

        for drug in drug_conc: 
            # --- Effetto lineare del farmaco sulla dissociazione corretta ---
            k_1_drug  = k_1_basal + alpha * drug

            # --- Effetto Hill del farmaco sulla associazione scorretta ---

            hill_fun  = (drug ** n) / (hill_costant**n + drug ** n) if drug > 0 else 0.0
            k_2_drug  = k_2 - hill_fun

            # --- Costruzione grafo diretto ---
            G = nx.DiGraph()
            G.add_nodes_from([1, 2, 3])
            G.add_weighted_edges_from([
                (1, 2, k1),       # associazione al sito corretto
                (2, 1, k_1_drug), # dissociazione dal sito corretto (drug-modulated)
                (1, 3, k2),  # associazione al sito errato (drug-modulated)
                (3, 1, k_2_drug),      # dissociazione dal sito errato
            ])

            # --- Matrix-Tree Theorem su grafo invertito ---
            G_rev = G.reverse()
            rho = {
                node: nx.number_of_spanning_trees(G_rev, root=node, weight="weight")
                for node in [1, 2, 3]
            }

            Z = sum(rho.values())
            if Z == 0:
                raise ValueError(
                    f"Z = 0 per params={param_set}, drug={drug}. "
                    "Verificare connettività e pesi del grafo."
                )

            drug_results.append({
                'drug':       drug,
                'hill_effect': round(hill_fun, 6),
                'k_1_drug':   round(k_1_drug, 6),
                'k2_drug':   round(k_2_drug, 6),
                'P1':         rho[1] / Z,
                'P2':         rho[2] / Z,
                'P3':         rho[3] / Z,
            })

        all_results.append({
            'params':  param_set,
            'results': drug_results,
        })

    return all_results