import numpy as np
import pandas as pd
from utils import *

def hybrid_sim(parameters, num_of_trials, expected_reward):
    """
    this funcation simulate an hybrid agent in the two step task 
    Args:
        parameters: parameters of the agent 
        num_of_trials: number of trials of the simulation
        reward_probs: a matrix 4*num_of_trials of the probability for reward of both second stage
    
    Returns:
        df: DataFrame of the behavior of the agent
        
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
    beta_2 = beta_1
    w = parameters['w']
    lambd = parameters['lambda']
    perseveration = 0
    
    # perseveration_array
    pers_array = np.zeros(2)
    
    # q_values of model free
    # first index - action; 
    # second index - state; 
    # third index - stage 
    q_mf = np.zeros(shape=(2 ,2, 2))
   
    # q_values of model based
    q_mb = np.zeros(2)
    
    # the weighted sum of model-based and model-free values
    q_net = np.zeros(2)
    
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
                    
        # calc prob with softmax for first stage
        prob_1 = np.exp( beta_1*(q_net+pers_array) ) / np.sum( np.exp( beta_1*(q_net+pers_array) ) ) 
        
        # choose action_1 according to prob for first stage
        action_1 = np.random.choice([0,1] , p=prob_1)
        
        # transation to second stage 
        transition_type = np.random.choice([0,1], p = [0.7, 0.3]) # 0 = common / 1 = rare
        state = state_transition_mat[action_1,transition_type]
        
        # calc prob with softmax for second stage
        prob_2 = np.exp( beta_2*(q_mf[:,state,1]) ) / np.sum( np.exp(beta_2*(q_mf[:,state,1]) ) ) 
        
        # choose action_2 according to prob for second stage
        action_2 = np.random.choice([0,1] , p=prob_2)

        # check if the trial is rewarded
        reward = np.random.choice([0,1] , p=[1-expected_reward[2*state+action_2, t],\
                                                                   expected_reward[2*state+action_2, t]])                
        # prediction error
        p_e_1 = q_mf[action_2,state,1] - q_mf[action_1,0,0] 
        p_e_2 = reward - q_mf[action_2,state,1] 
        
        # update q_mf according to q_learning formula
        q_mf[action_1,0,0] = q_mf[action_1,0,0] + alpha_1*p_e_1 + lambd*(alpha_1*p_e_2) 
        q_mf[action_2,state,1] = q_mf[action_2,state,1] + alpha_2*p_e_2
        
        # stroe data of the trial
        data.n_trial[t] = t  
        data.action_1_list[t] = action_1
        data.stage_2_state[t] = state
        data.transition_list[t] = transition_type
        data.action_2_list[t] = action_2
        data.reward_list[t] = reward
        data.probs_action_0[t] = prob_1[0]
        data.delta_q[t] = q_net[0] - q_net[1]
        
    df = pd.DataFrame(data.createDic())
    return df