import numpy as np
import pandas as pd
from utils import *

import numpy as np
import pandas as pd
from utils import *

def kdh_sim(parameters, num_of_trials, expected_reward):
    """
    this funcation simulate an k-Dominant Hand agent in the two step task 
    Args:
        param: parameters of the agent 
        num_of_trials: number of trials of the simulation
        transition_prob: a matrix 2*2 of the transition function from the first stage to the second stage 
        reward_probs: a matrix 4*num_of_trials of the probability for reward of both second stage 
                      states for all trials
    
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
    p_0 = parameters['p_0']
    p_1 = parameters['p_1']
    p_2 = parameters['p_2']
    p_3 = parameters['p_3']

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
        
        # probability of selecting action 0 in first stage 
        p = p_0 if t%2==0 else p_1

        # choose action according to the probability of the first stage
        action_1 = np.random.choice([0,1] , p=[p, 1-p])
                
        # sample a transation type
        transition_type = np.random.choice([0,1], p = [0.7, 0.3]) # 0 = common / 1 = rare
        # transation to second stage according to action and transation type
        state = state_transition_mat[action_1,transition_type]

        p_state = p_2 if state==0 else p_3
        
        # choose action according to the probability of the second stage
        action_2 = np.random.choice([0,1] , p=[p_state, 1-p_state])

        # check if the trial is rewarded
        reward = np.random.choice([0,1] , p=[1-expected_reward[2*state+action_2, t],\
                                                                   expected_reward[2*state+action_2, t]]) 
  
        # stroe data of the trial
        data.n_trial[t] = t  
        data.action_1_list[t] = action_1
        data.stage_2_state[t] = state
        data.transition_list[t] = transition_type
        data.action_2_list[t] = action_2
        data.reward_list[t] = reward
        data.probs_action_0[t] = p
      
    df = pd.DataFrame(data.createDic())
    return df
