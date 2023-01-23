import numpy as np
import pandas as pd
from utils import *
from scipy.optimize import minimize

def habit_fit(df,num_of_parameters_to_recover=3):
    """
    this funcation performs parameters recovery of habitual agent on the two-stage task  
    Args:
        df: DataFrame of the behavior of the agent in the simulation
    Returns:
        res: results of the minimize funcation 
    """
    # sample initial guess of the parameters to recover 
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    initial_guess[2] = np.random.uniform(0.1,10)
    # set bounds to the recover parameters 
    bounds = [(0,1),(0,1),(0.1,10)]
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
    num_of_trials = len(df)
    p_choice_1 = np.zeros(num_of_trials)
    p_choice_2 = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_stage_1 = list(map(int, df['action_stage_1']))
    action_stage_2 = list(map(int, df['action_stage_2']))
    state_of_stage_2 = list(map(int, df['state_of_stage_2']))
    reward_list = list(map(int, df['reward'])) 

    # set up paramters for recovary    
    alpha_1 = parameters[0]
    alpha_2 = parameters[1]
    beta_1 = parameters[2]
    beta_2 = beta_1
    
    # initialize Habit matrix
    H = np.zeros(2) + .5
    H_s_two = np.zeros(shape=(2,2)) + .5 

    for t in range(num_of_trials):  
        
        # calc prob with softmax for first stage
        p = np.exp(beta_1*(H)) / np.sum(np.exp(beta_1*H)) 
        
        # get true first action
        action_1 = action_stage_1[t]
        p_choice_1[t] = p[action_1]
        
        # update abit strengths for first stage
        H = (1-alpha_1)*H
        H[action_1] = H[action_1] + alpha_1
        
        state = state_of_stage_2[t]
    
        # calc prob with softmax for second stage
        p = np.exp(beta_2*(H_s_two[state])) / np.sum(np.exp(beta_2*H_s_two[state])) 

        # get true second action
        action_2 = action_stage_2[t]
        p_choice_2[t] = p[action_2]
        
        # updated habit strengths for second stage
        H_s_two[state] = (1-alpha_2)*H_s_two[state]
        H_s_two[state,action_2] = H_s_two[state,action_2] + alpha_2
  
    eps = 1e-10
    log_loss = -(np.sum(np.log(p_choice_1 + eps))+np.sum(np.log(p_choice_2 + eps)))
    return log_loss

