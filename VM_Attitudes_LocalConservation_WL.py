# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:14:46 2017

@author: William Lawrie V
"""
from pylab import *
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage

class cluster():
    
    def __init__(self,init_x,init_y,val,grid_x,grid_y,grid):
        
        self.grid_x  = grid_x
        self.grid_y  = grid_y
        self.grid    = grid    
        
        self.init_x  = init_x
        self.init_y  = init_y
        
        self.val   = val
        
        self.coords    = [[init_x],[init_y]]
        self.area      = 1
        self.perimiter = 1
        
        self.open_set =   []
        self.closed_set = []
        
    def create_coord_list(self):
        
        self.open_set = [[self.init_x,self.init_y]]
        self.open_x   = [self.init_x] 
        self.open_y   = [self.init_y]
        self.closed_set = []
        
        while True:
            self.add_neighbours_in_cluster(self.open_set[0][0],self.open_set[0][1])
            self.closed_set.append([self.open_set[0][0],self.open_set[0][1]])
            del self.open_set[0]
            if self.open_set == [] :
                self.coords = self.closed_set
                break
                
        
    def add_neighbours_in_cluster(self,x,y):
        nn = [[0,1],[0,-1],[1,0],[-1,0]]
        for i in range(4):
            if self.grid[(x+nn[i][0])%self.grid_x,(y+nn[i][1])%self.grid_y]==self.val and not ([(x+nn[i][0])%self.grid_x,(y+nn[i][1])%self.grid_y] in self.closed_set)and not ([(x+nn[i][0])%self.grid_x,(y+nn[i][1])%self.grid_y] in self.open_set):
                    self.open_set.append([(x+nn[i][0])%self.grid_x,(y+nn[i][1])%self.grid_y])
    
    def calculate_perimeter(self):
        perim = 0
        for i in range(np.size(self.coords,axis = 0)):
            perim_check = self.perim_check(self.coords[i][0],self.coords[i][1])
            if perim_check==1:
                perim+=1
        self.perimiter = perim
        
    def perim_check(self,x,y):
         perim = 0
         nn = [[0,1],[0,-1],[1,0],[-1,0]]        
         for i in range(4):
             if self.grid[(x+nn[i][0])%self.grid_x,(y+nn[i][1])%self.grid_y]!=self.val:
                 perim = 1
         return perim
     
    def calculate_area(self):
         self.area = np.size(self.coords,axis = 0)



class grid():
    
    def __init__(self,length,width,timesteps,adj,init):
        
        self.length = length
        self.width  = width                                                     #Dimensions of the grid
        self.timesteps = timesteps                                              #Number of timesteps for which to set aside memory
        self.magnetization = np.zeros(timesteps)                                #Stores concensus of grid at each time t
        self.DOI = np.zeros(timesteps)                                          #Stores density of ACTIVE interfaces
        self.DOIAttitudes = np.zeros(timesteps)                                 #Stores the density of ACTIVE attitudes at a given timestep
        self.opinions  = np.random.choice([-1,1],size=(length,width))           #Starting opinions of agents at the table
        self.attitudes = init*np.ones((length,width,4))                         #Starting attitudes of agents with other agents. init is the starting attitude
        self.time_array = np.zeros(timesteps)                                   #Stores an array of times to plot later
        self.attitude_adjust = adj                                              #Additive Value an opinion updates by on unsuccessful/sucessful event        
        self.clusterSizes = []
    
    def set_time(self,t):
        '''
        Updates the time array to the current time t
        
        Keyword Arguments:
            t - Current timestep
        
        Returns:
            nothing
        '''
        self.time_array[t] = t
        
    def set_mag(self,t):
        '''
        Updates the total magnitization of the system in the matrix
        self.magnetization for the current timestep t
        
        Keyword Arguments:
            t - Current timestep
        
        Returns:
            nothing
        '''
        self.magnetization[t] = np.sum(self.opinions)/(self.length*self.width) 
        
    def update_opinion(self):
        '''
        Updates the opinions and attitudes of the system of each agent for a 
        single timestep.
        
        Keyword Arguments:
            none
        
        Returns:
            nothing
        '''
        l_ran =  np.arange(self.length)
        w_ran =   np.arange(self.width)
        np.random.shuffle(l_ran)
        np.random.shuffle(w_ran)
        to_choose=[[1,0],[-1,0],[0,1],[0,-1]] # Down, Up, Right, Left
        for i in range(0,self.width):
            for j in range(0,self.length):
                rand_choice=np.random.choice(np.arange(4))
                flip_prob = self.calc_flip_prob(l_ran[i],w_ran[j],rand_choice)
                r = np.random.rand() 
                if r<flip_prob and flip_prob != -1:
                    #self.increase_attitude(l_ran[i],w_ran[j],rand_choice)
                    temp = self.opinions[l_ran[i],w_ran[j]]
                    self.opinions[l_ran[i],w_ran[j]]=self.opinions[(l_ran[i]+to_choose[rand_choice][0])%self.length,(w_ran[j]+to_choose[rand_choice][1])%self.width]
                    self.opinions[(l_ran[i]+to_choose[rand_choice][0])%self.length,(w_ran[j]+to_choose[rand_choice][1])%self.width] = temp        
                elif r>flip_prob and flip_prob==-1:
                    self.attitudes[l_ran[i],w_ran[j],rand_choice] += 0.000
                    #self.increase_attitude(l_ran[i],w_ran[j],rand_choice)
                elif r>=flip_prob and flip_prob!=-1:
                #elif r>=flip_prob:
                    self.decrease_attitude(l_ran[i],w_ran[j],rand_choice)
                    
                
    def calc_flip_prob(self,x,y,nn):
        to_choose=np.array([[1,0],[-1,0],[0,1],[0,-1]]) # Down, Up, Right, Left
        rand_x=to_choose[nn,0]
        rand_y=to_choose[nn,1]
        
        if self.opinions[x,y]==self.opinions[(x+rand_x)%self.length,(y+rand_y)%self.width]:
            return -1
        elif self.opinions[x,y]!=self.opinions[(x+rand_x)%self.length,(y+rand_y)%self.width]:
            rand2=(nn+np.random.choice([1,2,3]))%4
            rand_x2 = to_choose[rand2,0]
            rand_y2 = to_choose[rand2,1]
            
            if nn==0 or nn==2:
                rand3 = (nn+np.random.choice([2,3,4]))%4
            elif nn==1 or nn==3:
                rand3 = (nn+np.random.choice([1,2,4]))%4
            rand_x3 = to_choose[rand3,0]
            rand_y3 = to_choose[rand3,1]
            
            if self.opinions[x,y] != self.opinions[(x+rand_x2)%self.length,(y+rand_y2)%self.width] and self.opinions[(x+rand_x)%self.length,(y+rand_y)%self.width]!=self.opinions[(x+rand_x+rand_x3)%self.length,(y+rand_y+rand_y3)%self.width]:
                return self.attitudes[x,y,nn]
            
        return 0.0    
        
            
            
    def increase_attitude(self,x,y,direction):
        #self.attitudes[x,y,direction] += (1-self.attitude_adjust)*(np.abs(1-self.attitudes[x,y,direction]))
        if self.attitudes[x,y,direction] < 1.0 and self.attitudes[x,y,direction]+self.attitude_adjust<1.0:
            self.attitudes[x,y,direction]+=self.attitude_adjust
        elif self.attitudes[x,y,direction] < 1.0 and self.attitudes[x,y,direction]+self.attitude_adjust>1.0:
            self.attitudes[x,y,direction] = 1.0
        
        
        
    def decrease_attitude(self,x,y,direction):
        #self.attitudes[x,y,direction] -= (1-self.attitude_adjust)*self.attitudes[x,y,direction]
        if self.attitudes[x,y,direction]>0.0 and self.attitudes[x,y,direction]-self.attitude_adjust>0.0:
            self.attitudes[x,y,direction]-=self.attitude_adjust
        elif self.attitudes[x,y,direction]>0.0 and self.attitudes[x,y,direction]-self.attitude_adjust<0.0:
            self.attitudes[x,y,direction]=0.0
        
    def initialize_grid_ordered(self,radius):
        for i in range(0,self.length):
            for j in range(0,self.width):
                ic = np.absolute((self.length/2)-i)
                jc = np.absolute((self.width/2)-j)
                self.opinions[i,j]=-1
                if((ic**2+jc**2)**0.5 < radius):
                    self.opinions[i,j]=1
                  
                
    def calculate_DOI(self,t):
        con = np.zeros((self.length,self.width,4))
        con[:,:,0] =  self.opinions+np.roll(self.opinions,-1,axis=0)
        con[:,:,1] =  self.opinions+np.roll(self.opinions,1,axis=0)
        con[:,:,2] =  self.opinions+np.roll(self.opinions,-1,axis=1)
        con[:,:,3] =  self.opinions+np.roll(self.opinions,1,axis=1)
        boolean = (con>-2) & (con<2)
        interface = np.float16(boolean)
        attitudes_to_sum = np.zeros_like(interface)
        for ii in range(4):
            attitudes_to_sum+=interface*self.attitudes
        self.DOI[t] = np.sum(interface)/(4.0*self.length*self.width)    
        self.DOIAttitudes[t] = np.sum(attitudes_to_sum)/(4.0*self.length*self.width) 
        
    def plot_grid(self):
        fig, ax = plt.subplots()

        data = self.opinions

        cax = ax.imshow(data, interpolation='nearest', cmap='Greys')
        ax.set_title('Opinions')
        cbar = fig.colorbar(cax, ticks=[-1, 1])
        plt.show()
        
    def plot_attitudes(self):
        fig, ax = plt.subplots()

        data = np.sum(self.attitudes/4.0,axis=2)

        cax = ax.imshow(data, interpolation='nearest')
        ax.set_title('Attitudes')
        cbar = fig.colorbar(cax)
        plt.show()
    
    def plot_ConAndDOI(self,t):
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(self.time_array[0:t],self.magnetization[0:t])
        plt.title('Consensus over time')
        plt.xlabel('Time step')
        plt.ylabel('Net Consensus')
        plt.subplot(2,1,2)    
        plt.plot(self.time_array[0:t],self.DOI[0:t])
        plt.title('Density of Interface over time')
        plt.xlabel('Time step')
        plt.ylabel('DOI')
        plt.show()
        
    def plot_DOAI(self,t):
        plt.figure(2)
        plt.plot(self.time_array[0:t],self.DOIAttitudes[0:t])
        plt.xlabel('Time step')
        plt.ylabel('Density of Attitudes at Active Interfaces')
        plt.show()        

    
length = 128
width  = 128
timesteps = 50000
radius = 16
init = 0.99
n_avs = 1
adjust = np.linspace(0.001, 0.01001, num=n_avs)
time2exit = np.zeros(n_avs)
mag_at_end = np.zeros(n_avs)



for i in range(0,n_avs):
    print(i)
    t=0
    the_grid = grid(length,width,timesteps,adjust[i],init)
    #the_grid.plot_grid()
    the_grid.calculate_DOI(t)
    #the_grid.initialize_grid_ordered(radius)
    #print(the_grid.opinions)

    while the_grid.DOIAttitudes[t] > 0.0:
    #while t < timesteps-1:
        t+=1
        the_grid.update_opinion()
        the_grid.set_time(t)
        the_grid.set_mag(t)
        the_grid.calculate_DOI(t)
        #the_grid.plot_attitudes()
        #print(np.sum(the_grid.DOIAttitudes[t]))
        #print(np.sum(the_grid.attitudes))
        if(np.mod(100*t/timesteps,1)==0):
            #print('Program is: ',100*t/timesteps,'Percent complete...' )
            the_grid.plot_grid()
            the_grid.plot_attitudes()
            print(t,np.sum(the_grid.DOIAttitudes[t]),np.sum(the_grid.attitudes)/(4.0*length*width))
        
        
        
    mag_at_end[i] = np.abs(the_grid.magnetization[t])
    time2exit[i] = t


'''
CLUSTER CALCULATION
'''
closed_queue = []
cluster_counter = 0
cluster_areas = []
cluster_perimiters = []

for i in range(length):
    for j in range(width):
        if (not [i,j] in closed_queue):
            a_cluster = cluster(i,j,the_grid.opinions[i,j],length,width,the_grid.opinions)
            a_cluster.create_coord_list()
            a_cluster.calculate_area()
            a_cluster.calculate_perimeter()
            
            closed_queue+=a_cluster.coords
            cluster_areas.append( a_cluster.area)
            cluster_perimiters.append(a_cluster.perimiter)
            cluster_counter += 1
            
plt.figure(7)
plt.loglog(cluster_areas,cluster_perimiters,linestyle = 'none',marker='o')
plt.xlabel('Area')
plt.ylabel('Perimiter')
plt.show()




"""
The REST
"""
the_grid.plot_ConAndDOI(t)
the_grid.plot_DOAI(t)    

#np.savetxt(str(length)+'x'+str(width)+'probp0p99'+str(n_avs)+'avs.txt', mag_at_end)
plt.figure(6)
plt.hist(the_grid.clusterSizes,bins='auto')
#plt.xlim(2,np.max(the_grid.clusterSizes))
plt.xlabel('Cluster Size')
plt.ylabel('frequency')
#np.savetxt(str(length)+'x'+str(width)+'probp0p99'+str(n_avs)+'avs.txt', mag_at_end)

plt.figure(3)
plt.hist(mag_at_end,bins='auto')
plt.xlim(0,1)
plt.xlabel('Magnetization at Co-Existance Condition')
plt.ylabel('frequency')
plt.show()

#np.savetxt('16x16Adjust_0_1', time2exit)

plt.figure(4)
plt.plot(adjust,time2exit)
plt.xlim(0,1)
plt.xlabel('Adjustment Paramter')
plt.ylabel('Exit Time')

plt.figure(5)
plt.plot(adjust,mag_at_end)
plt.xlim(0,1)
plt.xlabel('Adjustment Paramter')
plt.ylabel('Magnitization at Exit')
plt.show()