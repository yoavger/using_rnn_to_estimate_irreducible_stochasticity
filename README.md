# 
Using recurrent neural network to estimate irreducible stochasticity in human choice-behavior

## Background 
Theory-driven computational modeling allows estimation of latent cognitive variables.  
Nonetheless, studies have found that computational models can systematically fail to describe some individuals.  
Presumably two factors contribute to this shortcoming:  
1. Higher internal noise (stochasticity in behavior).
2. Model miss-specification. 

However, when measuring behavior of individuals on cognitive tasks, these two factors are entangled and therefore hard to dissociate.  
Here we examine the use of RNNs to disentangled this two factors.  

## study_1_simulation

- simulating agents, fitting the theoretical models, RNN and Logistic regression model
run the following notebook:
```
study_1_simulation/code/sim_fit_predict_agents.ipynb
```
- plotting Figures of study 1 
```
study_1_simulation/code/plots.ipynb
```
## study_2_emprical
- Fitting the theoretical hybrid model and RNN
```
study_2_emprical/code/fit_predict_agents.ipynb
```
- To run the bayesian analysis of study 2 
```
study_2_emprical/code/bayesian.ipynb
```
- plotting Figures of study 2 
```
study_2_emprical/code/plots.ipynb
```



## Dependencie
- numpy
- pandas
- matplolib
- seaborn
- sklearn
- scipy 
- tqdm
- torch
- pymc
- bambi

