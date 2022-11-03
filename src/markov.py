"""Markov models."""

import numpy as np
import math

class MarkovModel: # the class mechanism is used to create new types of
    # objects in python in which data can be stored. 
    """Representation of a Markov model."""

    init_probs: list[float] 
    trans: list[list[float]] # Rownumber gives the state of X_i-1, 
    # columnnumber gives the state of X_i. 

    def __init__(self, # __init__ is a so-called constructor. The 
    # constructor makes it easier to create objects of the type 
    # created using the class mechanism. When a constructor is used,
    # an object of the type MarkovModel can be created by calling 
    # MarkovModel() where a list initial probabilities and a list of
    # lists of transitional probabilities are given as arguments.
                 init_probs: list[float],
                 trans: list[list[float]]):
        """Create model from initial and transition probabilities."""
        # Sanity check...
        k = len(init_probs) # k is the number of possible states that 
        # can be initial states. 
        assert k == len(trans) # The number of possible initial states
        # should be equal to the number of possible states that can be
        # transitioned from.
        for row in trans: 
            assert k == len(row) # the number of states that can be 
            # transitioned to (given by len(row)) should be equal to 
            # the number of states that can be transitioned from (
            # otherwise one could e.g., transition to a state and then
            # not come any further, if it was not possible to transition
            # from the new state.)

        self.init_probs = init_probs
        self.trans = trans

# The different states are given a number. The number corresponds to 
# the idx at which the probability of that state being the initial
# state is found in the init_probs list. And it corresponds to the 
# idx of the row list that gives the probabilities of making a 
# transition from that state in the trans list of lists. 
# And it corresponds to the idx of the column that gives the 
# probabilities of making a transition to that state in the trans list
# of lists.  

# Example of a Markov Model
SUNNY = 0
CLOUDY = 1

init_probs = [0.1, 0.9]
from_sunny = [0.3, 0.7]
from_cloudy = [0.4, 0.6]
trans = [
    from_sunny,
    from_cloudy
]

MM = MarkovModel(init_probs, trans)
#print(MM) # object represented as <something> since no __repr__(self)
# method used in the class definition. The __repr__(self) method is 
# used to make a textual representation of an object. 

# Access trans list of lists by MM.trans
# print(MM.trans)
# Access init_probs list by MM.init_probs
# print(MM.init_probs)

# tjek. 
def likelihood(x: list[int], mm: MarkovModel) -> float: # mm indeholder
    # de parameters, theta, der er givet ved Markov Modellen. 
    """
    Compute the likelihood of mm given x, i.e., P(x ; mm) where x is
    fixed.

    This is the same as the probability of x given mm,
    i.e., P(x ; mm), where mm is fixed.
    #>>> likelihood([0,0,0,1], MM) # version without log10.
    # 0.006299999999999999 # version without log10.
    >>> likelihood([0,0,0,1], MM) # version with log10.
    -2.2006594505464183
    """
    prob_initial_state = math.log10(mm.init_probs[x[0]])
    prob_X = prob_initial_state
    for i in range(1, len(x)): 
        prob_X += math.log10(mm.trans[x[i-1]][x[i]])
    # evt. nedenstående for at få P(X; mm) i stedet for log(P(X; mm)).
    # prob_X = 10**prob_X
    return prob_X

print(math.log10(0.1)+math.log10(0.3)+math.log10(0.3)+math.log10(0.7))
print(10**(-2.2006594505464183)) # 0.0063

def initial_state_prob(n: set, k: int) -> float:
    """
    Compute probability of state a being the initial state (initial
    state probability pi[a]).
    """
    M = len(n) # M number of sequences.
    counter = 0 # keep track of how many times state a is the initial
    # state.
    for i in range(M): # loop over the M sequences.
        counter += 1*(n[i][0]==k)
    return counter/M # return maximum-likelihood estimate. 

def transition_prob(n: set, total_transitions: int, k: int, h: int) -> float:
    """
    Compute probability of a given state being followed by another 
    given state (transition probability T[h, k]).
    """
    M = len(n) # M number of sequences.
    counter = 0 # keep track of how many times transition k -> h takes
    # place.
    for i in range(M): # loop over all sequences. 
        for j in range(1, len(n[i])): # loop over index i up to and 
            # including N-1 in sequence i of length N. 
            transitions += 1*(x[j-1]==k, x[j]==h)
    return transitions/total_transitions


def maximum_likelihood_estimates(n: set, s: int) -> tuple:
    """
    Returns parameters theta = (pi, T) for a Markov Model given a set
    of M sequences, {X_1, X_2,...,X_M}. 
    s is the number of possible states.
    >>> maximum_likelihood_estimates({[]})
    """
    # Estimate initial probabilities. 
    pi = [0]*s # indices 0, 1,..., k-1.
    M = len(x) # M sequences.
    for k in range(s): # loop over possible states.
        pi[k] = initial_state_prob(n, k)

    # Determine total number of transitions.
    transitions = 0
    for i in range(M): # loop over all sequences.
        transitions += len(n[i])-1

    # Estimate transition probabilities.
    T = [[0]*s]*s # [[0]*2]*2 = [[0,0]]*2 = [[0,0], [0,0]]
    for k in range(s): # loop over all possible from-states.
        for h in range(s): # loop over all possible to-states.
            T[k][h] = transition_prob(n, transitions, k, h)
    return (pi, T)
