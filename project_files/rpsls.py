# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:34:49 2017

@author: pc
"""


        
def rpsls(guess1, guess2):   
    
    result = (guess1 - guess2) % 5
    
    #print('Player 1 chooses', guess1)
    #print('Player 2 chooses', guess2)

    if result == 1 or result == 2:
        return 1 
    elif result == 3 or result == 4:
        return 2
    else:
        return 0
    
# test your code
#rpsls("rock")
#rpsls("Spock")
#rpsls("paper")
#rpsls("lizard")
#rpsls("scissors")