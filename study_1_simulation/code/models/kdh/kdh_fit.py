import numpy as np
import pandas as pd
from utils import *
from scipy.optimize import minimize

def kdh_fit(df,num_of_parameters_to_recover=4):
    """
    this funcation performs parameters recovery of k-Dominant Hand agent on the two-stage task  
    Args:
        df: DataFrame of the behavior of the agent in the simulation
    Returns:
        res: results of the minimize funcation 
    """
    # sample initial guess of the parameters to recover 
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    # set bounds to the recover parameters 
    bounds = [(0,1) for _ in range(num_of_parameters_to_recover)]

    res = minimize(
                    fun=parameters_recovary,
                    x0=initial_guess,
                    args=df,
                    bounds=bounds,
                    method='L-BFGS-B'
    )
    return res

def parameters_recovary(parameters, df):
    
    # objective to minimize
    log_loss = 0 
    num_of_trials = len(df)
    p_choice_1 = np.zeros(num_of_trials)
    p_choice_2 = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_stage_1 = list(map(int, df['action_stage_1']))
    action_stage_2 = list(map(int, df['action_stage_2']))
    state_of_stage_2 = list(map(int, df['state_of_stage_2']))
    reward_list = list(map(int, df['reward'])) 

    # set up paramters for recovary  
    p_0 = parameters[0]
    p_1 = parameters[1]
    p_2 = parameters[2]
    p_3 = parameters[3]
                    
    for t in range(num_of_trials):
        
        p = p_0 if t%2==0 else p_1
        
        p = [p, 1-p]        
        p_choice_1[t] = p[action_stage_1[t]]

        state = state_of_stage_2[t]
        p_state = p_2 if state==0 else p_3
        p_state = [p_state, 1-p_state]        
        p_choice_2[t] = p_state[action_stage_2[t]]

    eps = 1e-10
    log_loss = -(np.sum(np.log(p_choice_1 + eps))+np.sum(np.log(p_choice_2 + eps)))
    return log_loss