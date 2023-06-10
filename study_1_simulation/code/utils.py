import numpy as np
import pandas as pd

# utility funcation for configuration, simulation, storing 

# define BOUNDARY for random walk
LOWER_BOUNDARY = 0.2
UPPER_BOUNDARY = 0.8
DIFFUS_RATE = 0.025

def create_rndwlk(num_of_alternative, num_of_trials):
    # This funcation create reward probabilities 
    # according to random walk for each second stage choice
    init_probs = np.random.uniform(LOWER_BOUNDARY, UPPER_BOUNDARY, num_of_alternative)
    prob_list = np.zeros(shape=(num_of_alternative,num_of_trials))
    for n in range(num_of_trials):
        if n==0:
            for i in range(num_of_alternative): 
                prob_list[i,0] = init_probs[i]
        else:
            for i in range (num_of_alternative): 
                prob_list[i,n] = prob_list[i,n-1] + DIFFUS_RATE*np.random.normal()
                if prob_list[i,n] > UPPER_BOUNDARY : prob_list[i,n] = UPPER_BOUNDARY
                elif prob_list[i,n] < LOWER_BOUNDARY : prob_list[i,n] = LOWER_BOUNDARY
    return prob_list


def bce_loss(y_true ,y_predict):
    # binary cross entrophy loss with log base 2 
    return -( y_true * np.log2(y_predict) + (1-y_true)*np.log2(1 - y_predict) )

def configuration_parameters_hybrid():
    # 7 free parameters (α1, α2, β1, β2, w, λ, p)     
    parameters = {
                'alpha_1' : np.random.uniform(), # 0 <= alpha <= 1 of stage 1
                'alpha_2' : np.random.uniform(), # 0 <= alpha <= 1 of stage 2
                'beta_1' : np.random.uniform(0.1,10), # 0 <= beta <= 10  inverse temperature beta of stage 1
                'beta_2' : np.random.uniform(0.1,10), # 0 <= beta <= 10 inverse temperature beta of stage 2
                'w' :  np.random.uniform(), # 0 <= w <= 1 mf - mb weight 
                'lambda' : np.random.uniform(), # 0<= lambda <= 1 eligibility trace 
                'perseveration' : np.random.uniform()-0.5  # -0.5<= pers <= 0.5 perseveration / switching
    }
    return parameters

def configuration_parameters_habit():
    # 4 free parameters (α1, α2, β1, β2)     
    parameters = {
                'alpha_1' : np.random.uniform(), # 0 <= alpha <= 1 of stage 1
                'alpha_2' : np.random.uniform(), # 0 <= alpha <= 1 of stage 2
                'beta_1' : np.random.uniform(0.1,4), # 0 <= beta <= 10  inverse temperature beta of stage 1
                'beta_2' : np.random.uniform(0.1,4), # 0 <= beta <= 10 inverse temperature beta of stage 2
        }
    return parameters

def configuration_parameters_kdh():
    # k free parameters p_0 , p_1 , ... , p_k
    parameters = {
                'p_0' : np.random.uniform(),
                'p_1' : np.random.uniform(),
                'p_2' : np.random.uniform(),
                'p_3' : np.random.uniform()

    }
    return parameters


class DataOfSim():
# this class stores all the data of one simulation
# storing the following: action_1, stage_2_state, transation_type, action_2, reward

    def __init__ (self , num_of_trials):
        self.n_trial =  np.zeros(num_of_trials,dtype=int)
        self.action_1_list = np.zeros(num_of_trials)
        self.stage_2_state = np.zeros(num_of_trials)
        self.transition_list = ['' for i in range(num_of_trials)]
        self.action_2_list = np.zeros(num_of_trials)
        self.reward_list = np.zeros(num_of_trials)
        self.probs_action_0 = np.zeros(num_of_trials,dtype=np.float32)
        self.delta_q = np.zeros(num_of_trials,dtype=np.float32)

    def createDic(self):
        dic = {
                'n_trial':self.n_trial,
                'action_stage_1' : self.action_1_list,
                'state_of_stage_2' : self.stage_2_state,
                'transition_type' : self.transition_list,
                'action_stage_2' : self.action_2_list, 
                'reward' : self.reward_list,
                'probs_action_0' : self.probs_action_0,
                'delta_q': self.delta_q   
            }
        return dic 
    
    
