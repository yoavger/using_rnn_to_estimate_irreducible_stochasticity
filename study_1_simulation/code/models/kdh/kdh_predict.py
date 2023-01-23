import numpy as np
import pandas as pd

def kdh_predict(df,parameters):
    """
    this funcation predict the action of a recoverd k-Dominant Hand agent in the two step task 
    Args:
        paramters: parameters of the agent 
        df: DataFrame of the true behavior of the agent in the simulation we want to predict 
        
    Returns:
    accuracy - number of action predicted correctly (argmax) 
    p_choice_0 - a vector of length num_of_trials of the probability of choosing action 0 in the first stage
       
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

    # set up paramters 
    p_0 = parameters[0]
    p_1 = parameters[1]
    p_2 = parameters[2]
    
    for t in range(num_of_trials):
        p = p_0
        p = [p_0, 1-p_0]

        # predict action according max probs 
        action_1_predict = np.argmax(p)
        p_choice_1[t] = p[0]
        
        state = state_of_stage_2[t]

        p_state = p_1 if state==0 else p_2
        p_state = [p_state, 1-p_state]
        p_choice_2[t] = p_state[0]
        
        # cheek if prediction match the true action
        if action_1_predict == action_stage_1[t]:
            accuracy+=1
            
    return accuracy , p_choice_1 , p_choice_2 
    