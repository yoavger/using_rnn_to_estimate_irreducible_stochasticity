import numpy as np
import pandas as pd
from utils import *

import numpy as np
import pandas as pd
from utils import *

def habit_predict(df,parameters):
    """
    this funcation predict the action of a recoverd habitual agent in the two step task 
    Args:
        
        df: DataFrame of the true behavior of the agent in the simulation we want to predict 
         paramters: parameters of the agent
    Returns:
    accuracy - number of action predicted correctly (argmax) 
    choices_probs_0 - a vector of length num_of_trials of the probability of choosing action 0 in the first stage
        
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

    # set up paramters for recovary    
    alpha_1 = parameters[0]
    alpha_2 = parameters[1]
    beta_1 = parameters[2]
    beta_2 = beta_1

    # initialize Habits matrix
    H = np.zeros(2) + .5
    H_s_two = np.zeros(shape=(2,2)) + .5 

    for t in range(num_of_trials):   
        
        p = np.exp(beta_1*(H)) / np.sum(np.exp(beta_1*H)) 
               
        # predict action according max probs 
        action_1_predict = np.argmax(p)
        p_choice_1[t] = p[0]

        # get true action
        action_1 = action_stage_1[t]
        
        H = (1-alpha_1)*H
        H[action_1] = H[action_1] + alpha_1
        
        state = state_of_stage_2[t]
    
        # calc prob with softmax for second stage
        p = np.exp(beta_2*(H_s_two[state])) / np.sum(np.exp(beta_2*H_s_two[state])) 

        # get true second action
        action_2 = action_stage_2[t]
        p_choice_2[t] = p[0]
        
        # updated habit strengths for first stage
        H_s_two[state] = (1-alpha_2)*H_s_two[state]
        H_s_two[state,action_2] = H_s_two[state,action_2] + alpha_2
        

        # cheek if prediction match the true action
        if action_1_predict == action_1:
            accuracy+=1
            
    return accuracy , p_choice_1 , p_choice_2 