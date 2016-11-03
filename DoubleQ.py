import blackjack
from pylab import *
import numpy as np
import random

numEvaluationEpisodes = 1000000
numTrainingEpisodes = 1000000

alpha = 0.001
eps = 0.01
gamma = 1

Q1 = 0.000001 * np.random.rand(181,2) # NumPy array of correct size
Q2 = 0.000001 * np.random.rand(181,2) # NumPy array of correct size
Q1[False, : ] = Q2[False, :] = 0

def GreedyPolicy(s):     # use to print the policy
     a = argmax(Q[s,:])
     return a

def learn(alpha, eps, numTrainingEpisodes):
     global Q
     returnSum = 0.0
     for episodeNum in range(numTrainingEpisodes):
          G = 0
          # Fill in Q1 and Q2 
          # initialize the value for the s and a, do this firstly because if not, it will cause all results equal to 0 for some reason(skip the loop)
          
          greedy = random.random() # use to determine whether go greedy policy or random policy
          s = blackjack.init()
          if greedy < eps:
               a = random.randint(0,1)
          else:
               Q = Q1 + Q2
               a = argmax(Q[s, :])
     
          r, next_s = blackjack.sample(s, a)      
          # with 0.5 probability to go each
          prob = random.randint(0,1)          
          if prob: 
               Q1[s, a] = Q1[s, a] + alpha * (r + gamma * Q2[next_s, argmax(Q1[next_s, :])] - Q1[s, a])
          else:
               Q2[s, a] = Q2[s, a] + alpha * (r + gamma * Q1[next_s, argmax(Q2[next_s, :])] - Q2[s, a])
          G += r
          s = next_s          
          while s != False:
               if greedy < eps:
                    a = random.randint(0,1)
               else:
                    Q = Q1 + Q2
                    a = argmax(Q[s, :])
               r, next_s = blackjack.sample(s, a)
               # Show result (debug)
               #print("r: ", r, "s: ", next_s)
     
               # with 0.5 probability to go each
               prob = random.randint(0,1)     
               if prob: 
                    Q1[s, a] = Q1[s, a] + alpha * (r + gamma * Q2[next_s, argmax(Q1[next_s, :])] - Q1[s, a])
               else:
                    Q2[s, a] = Q2[s, a] + alpha * (r + gamma * Q1[next_s, argmax(Q2[next_s, :])] - Q2[s, a])
               s = next_s
               G += r          

          #print("Episode: ", episodeNum, "Return: ", G)
          returnSum = returnSum + G
          if episodeNum % 10000 == 0 and episodeNum != 0:
               print("Average return so far: ", returnSum/episodeNum)
     print(blackjack.printPolicy(GreedyPolicy))
           

     

def evaluate(numEvaluationEpisodes):
     returnSum = 0.0 
     for episodeNum in range(numEvaluationEpisodes):
          G = 0
          s = blackjack.init()
          # deterministic policy
          a = argmax(Q[s, :])
          r, s = blackjack.sample(s, a)
          G += r
     
          # get the r until get to the terminal s(S == False)
          while s != False:
               a = argmax(Q[s, :])
               r, s = blackjack.sample(s, a)
               G += r
     
        #  print("Episode: ", episodeNum, "Return: ", G)
          returnSum = returnSum + G

     return returnSum/numEvaluationEpisodes     

learn(alpha, eps, numTrainingEpisodes)
print (evaluate(numEvaluationEpisodes))
