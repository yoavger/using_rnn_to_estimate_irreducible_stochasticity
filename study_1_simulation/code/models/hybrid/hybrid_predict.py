import numpy as np
import pandas as pd
from utils import *

def hybrid_predict(df,parameters):
    """
    this funcation predict the action of a recoverd hybrid agent in the two step task 
    Args:
        paramters: parameters of the agent 
        df: DataFrame of the true behavior of the agent in the simulation we want to predict 
        
    
    Returns:
    accuracy - number of action predicted correctly (argmax) 
    p_choice_1 - a vector of length num_of_trials of the probability of choosing action 0 in the first stage
        
    action are coded: 
        0 and 1 
    state are coded: 
        0 - first stage
        1 - second stage first state
        2 - second stage second state
    """
     # counter of the number of action classified correctly (accuracy)
    accuracy = 0 
    num_of_trials = len(df)
    p_choice_1 = np.zeros(num_of_trials)
    p_choice_2 = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_stage_1 = list(map(int, df['action_stage_1']))
    action_stage_2 = list(map(int, df['action_stage_2']))
    state_of_stage_2 = list(map(int, df['state_of_stage_2']))
    reward_list = list(map(int, df['reward'])) 
 
    # set up paramters of the agent         
    alpha_1 = parameters[0]
    alpha_2 = parameters[1]
    beta_1 = parameters[2]
    beta_2 = beta_1
    w = parameters[3]
    lambd = parameters[4]
    perseveration = 0

    # initialize Q-values 
    q_mf = np.zeros(shape=(2,2,2))
    q_mb = np.zeros(2)
    q_net = np.zeros(2)
    pers_array = np.zeros(2)

    # state transition structure common/rare
    transition_prob = np.array(
                                [[.7,.3],
                                [.3,.7]]
                            )

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
        
        prob_1 = np.exp( beta_1*(q_net+pers_array) ) / np.sum( np.exp( beta_1*(q_net+pers_array))) 
        p_choice_1[t] = prob_1[0]
        action_1_predict = np.argmax(prob_1)
        
        action_1 = action_stage_1[t]
        state = state_of_stage_2[t]
        
        prob_2 = np.exp( beta_2*(q_mf[:,state,1]) ) / np.sum( np.exp(beta_2*(q_mf[:,state,1]) ) ) 
        p_choice_2[t] = prob_2[0]
        
        action_2_predict = np.argmax(prob_2)
        action_2 = action_stage_2[t]
    
        reward = reward_list[t]
    
        # prediction error 
        p_e_1 = q_mf[action_2,state,1] - q_mf[action_1,0,0] 
        p_e_2 = reward - q_mf[action_2,state,1] 
        
        # update q_mf according to q_learning formula
        q_mf[action_1,0,0] = q_mf[action_1,0,0] + alpha_1*p_e_1 + lambd*(alpha_1*p_e_2) 
        q_mf[action_2,state,1] = q_mf[action_2,state,1] + alpha_2*p_e_2

        if action_1_predict == action_1:
            accuracy+=1

    return accuracy , p_choice_1 , p_choice_2 
