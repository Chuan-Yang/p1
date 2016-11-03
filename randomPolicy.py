import blackjack
from pylab import *
import random 

def run(numEpisodes):
    returnSum = 0.0
    for episodeNum in range(numEpisodes):
        G = 0
        
        # initialize the value for the state and action
        state = blackjack.init()
        action = random.randint(0,1)
        reward, state = blackjack.sample(state, action)
        print("Reward: ", reward, "State: ", state)
        G += reward
        
        # get the reward until get to the terminal state(S == False)
        while state != False:
            action = random.randint(0,1)
            reward, state = blackjack.sample(state, action)
            # Show result (debug)
            print("Reward: ", reward, "State: ", state)
            G += reward
            
        print("Episode: ", episodeNum, "Return: ", G)
        returnSum = returnSum + G
    return returnSum/numEpisodes

# Test
print(run(1000))

