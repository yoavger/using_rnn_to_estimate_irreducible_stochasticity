import numpy as np
import pandas as pd
from utils import *
from scipy.optimize import minimize

def hybrid_fit(df,num_of_parameters_to_recover=5):
    """
    this funcation performs parameters recovery of hybrid agent on the two-stage task  
    Args:
        df: DataFrame of the behavior of the agent in the simulation
    Returns:
        res: results of the minimize funcation 
    """
    # sample initial guess of the parameters to recover 
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    initial_guess[2] = np.random.uniform(0.1,10)
    bounds = [(0,1),(0,1),(0.1,10),(0,1),(0,1)]
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

    q_mf = np.zeros(shape=(2,2,2))
    q_mb = np.zeros(2)
    q_net = np.zeros(2)
    pers_array = np.zeros(2)
   
    # state transition structure common/rare
    transition_prob = np.array(
                                [[.7,.3],
                                [.3,.7]]
                            )

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
    w = parameters[3]
    lambd = parameters[4]
    perseveration = 0

    for t in range(num_of_trials):
        
        # Q_model-based values of the first level actions (Bellman’s equation)
        q_mb[0] = (transition_prob[0,0]*np.max(q_mf[:,0,1])) + (transition_prob[0,1]*np.max(q_mf[:,1,1]))
        q_mb[1] = (transition_prob[1,0]*np.max(q_mf[:,0,1])) + (transition_prob[1,1]*np.max(q_mf[:,1,1]))

        # net action values at the first stage as the weighted sum of model-based and model-free values
        q_net[0] = (w*q_mb[0]) + ((1-w)*q_mf[0,0,0])  
        q_net[1] = (w*q_mb[1]) + ((1-w)*q_mf[1,0,0])
        
        # indicator of previous trial action for perseveration 
        if t > 0 : 
            pers_array[action_1] = perseveration
            pers_array[1-action_1] = 0
        
        # first choices
        action_1 = action_stage_1[t]
        p_choice_1[t] = np.exp(beta_1*(q_net[action_1] + pers_array[action_1])) / np.sum(np.exp(beta_1*(q_net+pers_array))) 
        
        state = state_of_stage_2[t]
        
        # second choices
        action_2 = action_stage_2[t]
        p_choice_2[t] = np.exp(beta_2 * q_mf[action_2,state,1]) / np.sum(np.exp(beta_2*q_mf[:,state,1]))
        
        reward = reward_list[t]
    
        # prediction error 
        p_e_1 = q_mf[action_2,state,1] - q_mf[action_1,0,0] 
        p_e_2 = reward - q_mf[action_2,state,1] 
        
        # update q_mf according to q_learning formula
        q_mf[action_1,0,0] = q_mf[action_1,0,0] + alpha_1*p_e_1 + lambd*(alpha_1*p_e_2) 
        q_mf[action_2,state,1] = q_mf[action_2,state,1] + alpha_2*p_e_2
        
    eps = 1e-10
    log_loss = -(np.sum(np.log(p_choice_1 + eps))+np.sum(np.log(p_choice_2 + eps)))
    return log_loss
