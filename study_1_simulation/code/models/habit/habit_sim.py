import numpy as np
import pandas as pd
from utils import *

def habit_sim(parameters, num_of_trials, expected_reward):
    """
    this funcation simulate an habitual agent in the two step task 
    Args:
        parameters: parameters of the agent 
        num_of_trials: number of trials of the simulation
        reward_probs: a matrix 4*num_of_trials of the probability for reward of both second stage
    
    Returns:
        df: DataFrame of the behavior of the agent in the simulation
        
    action are coded: 
        0 and 1 
    state are coded: 
        0 - first stage
        1 - second stage first state
        2 - second stage second state
    """
    # set up parameters 
    alpha_1 = parameters['alpha_1']
    alpha_2 = parameters['alpha_2']
    beta_1 = parameters['beta_1']
    beta_2 = parameters['beta_2']

    H = np.zeros(2) + .5 
    H_s_two = np.zeros(shape=(2,2)) + .5 
    
    # state transition structure common/rare
    transition_prob = np.array(
                            [[.7,.3],
                            [.3,.7]]
    )
    state_transition_mat = np.array(
                                [[0,1],
                                [1,0]]
    )
    # store data from each trial 
    data = DataOfSim(num_of_trials)

    for t in range(num_of_trials):

        # calc prob with softmax for first stage
        prob_1 = np.exp(beta_1*(H)) / np.sum(np.exp(beta_1*H)) 

        # choose action according to prob for first stage
        action_1 = np.random.choice([0,1] , p=prob_1)
        
        # updated habit strengths for first stage
        H = (1-alpha_1)*H
        H[action_1] = H[action_1] + alpha_1
        
        # sample a transation type
        transition_type = np.random.choice([0,1], p = [0.7, 0.3]) # 0 = common / 1 = rare
        # transation to second stage according to action and transation type
        state = state_transition_mat[action_1,transition_type]
        
        # calc prob with softmax for second stage
        prob_2 = np.exp( beta_2*(H_s_two[state]) ) / np.sum( np.exp(beta_2*(H_s_two[state]) ) ) 
        
        # choose action_2 according to prob for second stage
        action_2 = np.random.choice([0,1] , p=prob_2)
        
        # check if the trial is rewarded
        reward = np.random.choice([0,1] , p=[1-expected_reward[2*state+action_2, t],\
                                                                   expected_reward[2*state+action_2, t]])   
        # updated habit strengths for second stage
        H_s_two[state] = (1-alpha_2)*H_s_two[state]
        H_s_two[state,action_2] = H_s_two[state,action_2] + alpha_2
        
        data.n_trial[t] = t  
        data.action_1_list[t] = action_1
        data.stage_2_state[t] = state
        data.transition_list[t] = transition_type
        data.action_2_list[t] = action_2
        data.reward_list[t] = reward
        data.probs_action_0[t] = prob_1[0]
        data.delta_q[t] = H[0] - H[1]
       
    df = pd.DataFrame(data.createDic())
    return df
