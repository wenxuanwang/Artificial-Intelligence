# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#
#THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING
#A TUTOR OR CODE WRITTEN BY OTHER STUDENTS - Wenxuan Wang 2165909


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"   
        GhostPos = successorGameState.getGhostPositions()
        newFoodList = newFood.asList()
        currentFoodList = currentFood.asList()
        myfood = []
        distToGhost = []      
        
        #nearest ghost
        for ghost in GhostPos:
            distToGhost.append(manhattanDistance(ghost,newPos))
        ghostDist = min(distToGhost)
        
        #if dist<5 then return negative infinity
        if ghostDist < 5:
            return -sys.maxint
        #next is food
        if len(newFoodList) < len(currentFoodList): 
            return sys.maxint 
        
        for food in newFood.asList():
            myfood.append(manhattanDistance(food,newPos))
        foodDist = min(myfood)        
        return -foodDist
        
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState,0,self.depth-1)[1]
    
    def minimax(self,gameState,agentIndex,depth):  
            
        if gameState.isWin() or gameState.isLose():    
            return (self.evaluationFunction(gameState), None)      
        if agentIndex == gameState.getNumAgents():    
            return self.minimax(gameState,0,depth-1) 
        
        return self.MiniMax(gameState,agentIndex,depth)      
      
    
    def MiniMax(self, gameState, agentIndex, depth):
        nextAction = None
        action = (0, None)  
        #ghost
        if(agentIndex >= 1):
            v = sys.maxint                    
            for actions in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1 and depth == 0:
                    value = (self.evaluationFunction(gameState.generateSuccessor(agentIndex,actions)), actions)
                else:
                    value = self.minimax(gameState.generateSuccessor( agentIndex,actions),agentIndex +1, depth) 
                if value[0]<v:
                    nextAction = actions
                    v=value[0]                    
                    action = (v,nextAction) 
        #pacman                                           
        else: 
            v = -sys.maxint    
            for actions in gameState.getLegalActions(agentIndex):
                value = self.minimax(gameState.generateSuccessor( agentIndex,actions),agentIndex+1,depth)
                if value[0] > v:         
                    nextAction = actions
                    v = value[0]
                    action = (v, nextAction)                          
        return action
    
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(gameState,0,-sys.maxint, sys.maxint,self.depth-1)[1]

    def alphabeta(self,gameState,agentIndex,alpha,beta,depth):
        if gameState.isWin() or gameState.isLose():      
            return (self.evaluationFunction(gameState),None)
        if agentIndex == gameState.getNumAgents():      
            return self.alphabeta(gameState,0,alpha,beta, depth-1)
        
        return self.AlphaBeta(gameState, agentIndex,alpha,beta,depth)
    
    def AlphaBeta(self, gameState,agentIndex,alpha,beta,depth):
        nextAction = None
        action = (0,None)
        #ghost
        if agentIndex >= 1:
            v = sys.maxint
            for actions in gameState.getLegalActions(agentIndex) :            
                if agentIndex == gameState.getNumAgents()- 1 and depth == 0:        
                    value = ( self.evaluationFunction(gameState.generateSuccessor(agentIndex,actions)),actions)
                else:
                    value = self.alphabeta(gameState.generateSuccessor(agentIndex, actions),agentIndex +1,alpha,beta,depth)
          
                if value[0] < v:
                    nextAction = actions
                    v = value[0]            
                    action = (v,nextAction)                    
                    if v < alpha:   return action          
                    beta = min(beta,v)            
        #pacman  
        else:
            v = -sys.maxint
            for actions in gameState.getLegalActions(agentIndex):
                value = self.alphabeta(gameState.generateSuccessor(agentIndex,actions),agentIndex+1,alpha,beta,depth)
                
                if value[0] > v:
                    nextAction = actions
                    v = value[0]
                    action = (v,nextAction)                    
                    if v > beta:  return action        
                    alpha = max(v , alpha)                            
        return action
    
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState,0, self.depth-1)[1]

    def expectimax(self,gameState, agentIndex, depth):
      
        if gameState.isWin() or gameState.isLose(): 
            return (self.evaluationFunction(gameState), None)
        if agentIndex == gameState.getNumAgents(): 
            return self.expectimax(gameState ,0,depth-1)
       
        return self.ExpectMax(gameState,agentIndex, depth)
           
    def ExpectMax(self,gameState, agentIndex, depth): 
        nextAction = None
        action = (0, None)
        #ghost
        if agentIndex >= 1:     
            exp = 0
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1 and depth == 0:
                    value=(self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)),action)[0]
                else:
                    value = self.expectimax(gameState.generateSuccessor(agentIndex, action),agentIndex +1,depth)[0]
                if type(value) is tuple:
                    value = value[0]
                exp += value * (1 / float ( len(gameState.getLegalActions(agentIndex)) ) )
                action= (exp, None)
            return action  
        #pacman      
        else:
            v = -sys.maxint
            for actions in gameState.getLegalActions(agentIndex):
                value = self.expectimax(gameState.generateSuccessor(agentIndex,actions),agentIndex +1,depth)
                if value > v:
                    nextAction = actions
                    v = value                    
                    action = (v ,nextAction)
        return action
    
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] 
    closestGhost = sys.maxint
    closestFood = sys.maxint
    closestCapsule = sys.maxint
    
    score = currentGameState.getScore()   
    for i in range(len(newGhostStates)):
        distToGhost = manhattanDistance(newPos,currentGameState.getGhostPosition(i + 1))
        if distToGhost < closestGhost: 
            closestGhost = distToGhost
    closestGhost = sys.maxint if closestGhost == 0 else closestGhost
    
    for food in newFood.asList():
        closestFood = min(closestFood, manhattanDistance(food, newPos))        
    closestFood = sys.maxint if closestFood == 0 else closestFood
    
    for capsules in currentGameState.getCapsules():
        closestCapsule = min(closestCapsule, manhattanDistance(capsules, newPos))        
    closestCapsule = sys.maxint if closestCapsule == 0 else closestCapsule    
      
    score += sum(time for time in newScaredTimes)
    
    return score + 10.0/ closestGhost + 10.0/closestFood + 10.0/closestCapsule

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

