import sys
from contextlib import closing
from io import StringIO
from typing import Optional
import numpy as np
from gym import Env, spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation 
import itertools as tools
from tabulate import tabulate
from __future__ import print_function
from IPython import display
from math import comb
from random import randint

class LawnMowing(gym.Env):
    """
    The Lawn Mowing Problem that we have:
    
    ### Description
    We will assume here that we have a robot designated to perform lawn mowing task to a certain area.
    For this scenario, let's have 3 areas for now. However, I will try to make the code general such 
    that it adapts to whatever number of areas it is provided with. The provided map includes tuples that
    resembles the following:
    -Tuples that resembles the areas which contains the following information, respectively: the points
     that define the area, the corners of the area, the increment between the points along the x axis, and
     the increments between the points along the y axis.
    
    ### Actions
    Given "a = number of areas" and "p = number of possible patterns", there are [a+p+1] discrete deterministic 
    actions:
    - M_i : Move from current area to area i
    - SLM_j : Start lawn mowing in the current area following pattern j
    - RtO : Return to originial location
    Also, the actions will be in the following order:
    - [0:a-1]: Moving actions to all the possible areas
    - [a:a+p-1]: The lawn mowing patterns available to us
    - [last element]: RtO action
    
    ### Observations
    The number of possible states will be: ((a+1)*sum([aCr: r=0:a])), where "a: number of areas" and the sum will be
    the possibile combinations of the areas. All these states are reachable. The episode ends when "Done = 1". 
    
    Current area could be:
    0:Not a given area
    1:First area
    .
    .
    .
    a:last area 
    ### Rewards
    The rewards are the same as the ones defined in the script.
    
    ### Rendering
    
    
    ### Arguments
    
    
    """

    metadata = {"render_modes": ["auto"], "render_fps": 4}

    def __init__(self):
        
        #Asking the user to input the map (areas to be lawn mowed):
        self.map = input("Please input the areas to be lawn mowed: ")
        #Creating a dictionary for the input areas:
        self.map_dict = dict({namestr(self.map[i],globals())[0]:self.map[i] for i in range(len(self.map))})
        #Asking the user to input the available patterns to be used:
        self.patterns = input("Please input the available patterns for lawn mowing: ")
        #Creating a dictionary for the input patterns:
        self.patterns_dict = dict({namestr(self.patterns[i],globals())[0]:self.patterns[i] for i in range(len(self.patterns))})
        #Asking the user to input its starting location:
        self.starting_loc = input("Please provide the starting the point: ")
        #Defining the path that will be followed though out the episode:
        self.path = self.starting_loc[None,:]
        
        #The number of areas that we have:
        num_areas = len(self.map)
        #The number of patterns that we have:
        num_patterns = len(self.patterns)
        #The number of states that we have:
        num_states = (num_areas+1)*comb(num_areas,a) for a in range(num_areas)
        #The number of actions that we have:
        num_actions = 1+num_areas+num_patterns

        #Encoding our areas as numbers:
        areas_as_numbers = list(range(1,num_areas+1))
        #All the possible states that we can have:
        Possible_States = []
        for CA in range(len(areas_as_numbers)+1):
            for i in range(len(areas_as_numbers)+1):
                for VA in tools.combinations(areas,i):
                    VA = list(VA)
                    NA = list(set(areas_as_numbers)-set(VA))
                    if set(areas_as_numbers) == set(VA):
                        Possible_States.append([CA,VA,NA,'Done = 1'])
                    else: Possible_States.append([CA,VA,NA,'Done = 0'])
        #Not sure what this is. I think it represents the probability of each state,action pair:
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in Possible_States
        }
        
        #In here, we need to define the states that we will be using and updating:
        
        #Current area (it is the starting point and we check whether it is in any area or not):
        self.CA = current_area(self.path[-1],self.map)
        #Visited areas initialized as an empty set (fresh start):
        self.VA = []
        #New areas which are all the given areas (fresh start): 
        self.NA = areas_as_numbers
        #"done" flag initialized to zero:
        done = 0
        #Given areas (In their numbers representation)
        given_areas = areas_as_numbers
        #The initial state:
        self.initial_state = tuple([self.CA,self.VA,self.NA,done])
        #Initializing a list to put the tasks in:
        self.Tasks = []
        #Initializing a variable to save the covered distance in:
        self.distance = 0
        
        for action in range(num_actions):
            #Note that the available actions for the agent are: M & SLM until the done flag is equal to one
            #at which the only available action will be RtO
            if action in range(1,num_areas+1):
                self.CA = action
                self.Tasks.append('Move to area '+self.map_dict.keys()[action-1])
                if self.CA in set(self.VA):
                    reward = -100
                else:
                    the_closest_corner = closest_corner(self.path[-1],self.map[action-1][1])
                    d = dist(the_closest_corner,self.path[-1])
                    reward = 20-(d**2)
                    self.path = np.insert(self.path,self.path.shape[0],the_closest_corner.tolist(),axis=0)
            if action in range(num_areas+1,num_areas+num_patterns+1):
                self.Tasks.append('Start Lawn Mowing using the pattern: '+self.patterns_dict.keys()[action-(num_areas+1)])
                if self.CA in set(self.VA):
                    reward = -100
                else:
                    pat_idx = action-(num_areas+1)
                    path_in_area, d = self.patterns[pat_idx](self.path[-1],self.map[self.CA-1])
                    reward = 20-(d**2)
                    self.path = np.insert(self.path,self.path.shape[0],path_in_area.tolist(),axis=0)
                    self.VA.append(self.CA)
                    self.NA = set(self.NA)-set(self.VA)
            if set(self.VA)==set(given_areas):
                action = 0
                done = 1
                d = dist(self.path[-1],self.path[0])
                reward = 20-(d**2)
                self.path = np.insert(self.path,self.path.shape[0],self.path[0].tolist(),axis=0)
                
            new_state = tuple([self.CA,self.VA,self.NA,done])
            self.distance = self.distance+d
            self.P[state][action].append((1.0,new_state,reward,done))
            if done == 1:
                break
        
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        
    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]
    
    def current_area(point,areas):
        for m in range(len(areas)):
            if (point[0][0] >= np.min(areas[m][1],axis=0)[0]) and (point[0][0] <= np.max(areas[m][1],axis=0)[0]) and (point[0][1] >= np.min(areas[m][1],axis=0)[1]) and (point[0][1] <= np.max(areas[m][1],axis=0)[1]):
                current_area = m+1
                break
            else: current_area = 0
        return current_area
    
    def closest_corner(p1,p2):
        starting_corner = p2[0] 
        for j in range(p2.shape[0]):
            if dist(p1,p2[j]) <= dist(p1,starting_corner):
                starting_corner = p2[j]
        return starting_corner
    
    def dist(p1,p2):
        return(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
    
    def categorical_sample(prob_n, np_random):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np_random.random()).argmax()


    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (show to plot a scatter plot for list in pythonhow to plot a scatter plot for list in python, r, d, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = self.initial_state
        self.lastaction = None
        if not return_info:
            return self.s
        else:
            return self.s, {"prob": 1}

    def render(self, mode="auto"):
        x_data = self.path[:,0]
        y_data = self.path[:,1]

        fig = plt.figure()

        lines = plt.plot([])
        line = lines[0]
        
        coord = [self[i][0].T for i in range(len(self.map))]
        
        def scatter_plot(list):
            x = []
            y = []
            colors = []
            labels = []
            counter = 0
            for i in list:
                x = i[0]
                y = i[1]
                rand_color = '#%06X' % randint(0, 0xFFFFFF)
                plt.scatter(x,y,label = 'area_'+chr(counter+65),color = rand_color)
                plt.legend()
                plt.show
                counter += 1

        def animation_plot(i):
            scatter_plot(coord)
            line.set_data((x_data[0:i],y_data[0:i]))

        anim = FuncAnimation(fig, animation_plot, frames=len(self.path)+1, interval=200)
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()