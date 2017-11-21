# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:29:34 2017

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
from itertools import compress
import time as thyme

class grid():
    '''
    The grid contains all information about the system, including the opinions
    of each agent/ whther a site is occupied or not, and the wealth of each 
    site. 
    '''
    def __init__(self, length,width,DOV,thresh):
        
        self.length = length
        self.width  = width
        
        self.DOI       = np.zeros(10000)
        self.DOI_size  = 1
        
        self.DOV       = DOV # Density of vacancies
        self.threshold = thresh #A threshold of 0.5 means a total weighting of zero like-minded neighbours
        
        self.opinions = np.random.choice([-1,0,1],size=(length,width),p = [(1.0-self.DOV)/2.0,self.DOV,(1.0-self.DOV)/2.0])
        self.n_agents = np.sum(np.abs(self.opinions))
        self.n_vac    = self.length*self.width-self.n_agents
        
        self.worth    = np.zeros((length,width))
        self.happiness= np.zeros((length,width))
        self.mag      = np.zeros((length,width))
        self.weights  = np.array(([0,1,0],[1,1,1],[0,1,0]))
        
        self.site_list = np.asarray(np.meshgrid(np.linspace(0,self.length-1,self.length),np.linspace(0,self.width-1,self.width)),dtype='int')
        self.vacant_sitelists = np.zeros((length,width))
        self.moves_avaliable_check = 1
        
    def calculate_global_happiness(self):
        #con = scipy.ndimage.filters.convolve(self.opinions,self.weights,mode='wrap')
        con_l = self.opinions*np.roll(self.opinions,1,axis=0)
        con_r = self.opinions*np.roll(self.opinions,-1,axis=0)
        con_u = self.opinions*np.roll(self.opinions,-1,axis=1)
        con_d = self.opinions*np.roll(self.opinions,1,axis=1)
        
        con = ((con_l+con_r+con_u+con_d)+4.0)/8.0                   #4.0 shifts to the positive
        self.mag = con                                                            #8.0 normalizes to (0=<H=<1)
        #print(con)
        self.happiness = (con>=self.threshold).astype(int)
        
    def calculate_global_worth(self):
        self.worth = np.abs(scipy.ndimage.filters.convolve(self.opinions,self.weights,mode='wrap'))/5.0        
        #print()
    
    
    def get_unhappy_agents(self):
        is_agent = (self.opinions != 0)
        agent_sitelist = self.site_list[:,np.asarray(is_agent,dtype='bool')]
        agent_coordlist = create_coords_list(agent_sitelist)
        
        is_unhappy = (self.happiness==0)
        unhappy_sitelist = self.site_list[:,np.asarray(is_unhappy,dtype='bool')] 
        unhappy_coordlist = create_coords_list(unhappy_sitelist)
        #unhappy_agents = get_intersection(unhappy_sitelist,agent_sitelist)
        unhappy_agents = [list(x) for x in set(tuple(x) for x in unhappy_coordlist).intersection(set(tuple(x) for x in agent_coordlist))]
        #self.plot_grid()
        #self.plot_happiness()
        return unhappy_agents
        #print(unhappy_agents)  # in col,row
    
    def update_site(self):
        
        unhappy_sitelist = self.get_unhappy_agents()  #for some reason.... in col row
        #print(self.happiness)
        #print('Únhappy Sites',unhappy_sitelist)
        #thyme.sleep(1)
        if np.size(unhappy_sitelist)>0:
            complete = 0
            while(complete==0):
                #print('Únhappy Sites',unhappy_sitelist)
                if np.size(unhappy_sitelist,axis=0)==0:
                    self.moves_avaliable_check = 0
                    break
                
                rand_choice = np.random.choice(np.arange(0,np.size(unhappy_sitelist,axis=0)))
                #print(rand_choice)
                c_x = unhappy_sitelist[rand_choice][1]
                c_y = unhappy_sitelist[rand_choice][0]
                coords = np.array([[unhappy_sitelist[rand_choice][1]],[unhappy_sitelist[rand_choice][0]]]) #retains [row,col] format
                #print('Chosen Unhappy Site:', coords)
                
                wealth = self.worth[coords[0],coords[1]]
                 
                is_vacant = (self.opinions == 0)
                #print('Possible Vacant Sites:', is_vacant)
                is_affordable = (self.worth <= wealth)
                #print('Possible Affordable Sites:', is_affordable)
                vacant_sitelist  = self.site_list[:,np.asarray(is_vacant,dtype='bool')]
                vacant_coordlist = create_coords_list(vacant_sitelist)
                #self.vacant_sitelists  = self.site_list[:,np.asarray(is_vacant,dtype='bool')]
                #print('Possible vacant Sites:',vacant_sitelist)
                #np.transpose(vacant_sitelist)
                affordable_sitelist  = self.site_list[:,np.asarray(is_affordable,dtype='bool')]
                affordable_coordlist = create_coords_list(affordable_sitelist)
                #print('Possible Affordable Sites:',affordable_sitelist)
                #print('Affordable AND Vacant sites: ',sitelist)
                #print(vacant_sitelist,affordable_sitelist,better)
                #print(self.opinions)
                #print(vacant_sitelist)
                
                #print('before',sitelist)
                #(np.size(sitelist,axis=0))
                better = self.local_happiness(c_x,c_y)
                better_coordlist = create_coords_list(better)
                #print(vacant_coordlist,affordable_coordlist,better_coordlist)
                #sitelist = get3_intersection(vacant_sitelist,affordable_sitelist,better)
                #print('Final Sitelist',sitelist)
                sitelist = [list(x) for x in (set(tuple(x) for x in vacant_coordlist).intersection(set(tuple(x) for x in affordable_coordlist))).intersection(set(tuple(x) for x in better_coordlist))]
                #print('Final Sitelist',sitelist)
                #new = [x for x in sitelist if better[x]==True]
                #sitelist = list(compress(sitelist,better))
                
                #print('áfter',new)
                #thyme.sleep(0.001)
                #print('Únhappy Sites',unhappy_sitelist)
                del unhappy_sitelist[rand_choice]
                #print('Únhappy Sites with deleted: ',rand_choice,unhappy_sitelist)
                
                    
                if sitelist:
#                    print('self.local_happiness(c_x,c_y,sitelist): ',self.local_happiness(c_x,c_y,sitelist))
#                    print('self.mag[c_x,c_y]',self.mag[c_x,c_y])
#                    print('Sites with higher utility: ', better)
#                    print('Final Sitelist',sitelist)
                    random_move = np.random.choice(np.arange(0,np.size(sitelist,axis=0)))
                    r_x = sitelist[random_move][1]
                    r_y = sitelist[random_move][0]
                    
                    temp = self.opinions[c_x,c_y]
                    self.opinions[c_x,c_y] = self.opinions[r_x,r_y]
                    self.opinions[r_x,r_y] = temp
                    complete = 1
    
#    def local_happiness(self,x,y,coords):
#        val = self.opinions[x,y]
#        #print(coords,(np.size(coords,axis=0)))
#        better=[]
#        for i in range(np.size(coords,axis=0)):
#            better.append((val*self.opinions[(coords[i][0]+1)%self.length,(coords[i][1]+0)%self.width]+val*self.opinions[(coords[i][0]-1)%self.length,(coords[i][1]+0)%self.width]+val*self.opinions[(coords[i][0]+0)%self.length,(coords[i][1]+1)%self.width]+val*self.opinions[(coords[i][0]+0)%self.length,(coords[i][1]-1)%self.width]+4.0)/8.0)
#        return better    
            
    
    def local_happiness(self,x,y):
        val = self.opinions[x,y]
        self.opinions[x,y] = 0
        is_better = ((val*self.opinions[(self.site_list[1,:,:]+1)%self.length,(self.site_list[0,:,:]+0)%self.width]+val*self.opinions[(self.site_list[1,:,:]-1)%self.length,(self.site_list[0,:,:]+0)%self.width]+val*self.opinions[(self.site_list[1,:,:]+0)%self.length,(self.site_list[0,:,:]+1)%self.width]+val*self.opinions[(self.site_list[1,:,:]+0)%self.length,(self.site_list[0,:,:]-1)%self.width]+4.0)/8.0>self.mag[x,y]) 
        self.opinions[x,y] = val
        #print('Determining better sitelist for coord',x,y)
        #self.plot_grid()
        #self.plot_happiness()
        
        #print((val*self.opinions[(self.site_list[0,:,:]+1)%self.length,(self.site_list[1,:,:]+0)%self.width]+val*self.opinions[(self.site_list[0,:,:]-1)%self.length,(self.site_list[1,:,:]+0)%self.width]+val*self.opinions[(self.site_list[0,:,:]+0)%self.length,(self.site_list[1,:,:]+1)%self.width]+val*self.opinions[(self.site_list[0,:,:]+0)%self.length,(self.site_list[1,:,:]-1)%self.width]+4.0)/8.0)
        #print(self.mag[x,y],is_better)
        better_sitelist = self.site_list[:,np.asarray(is_better,dtype='bool')]
        #print(better_sitelist)
        #better=[]
        return better_sitelist 
    
    def calculate_DOI(self,t):
        con = np.zeros((self.length,self.width,4))
        con[:,:,0] =  self.opinions*np.roll(self.opinions,-1,axis=0)
        con[:,:,1] =  self.opinions*np.roll(self.opinions,1,axis=0)
        con[:,:,2] =  self.opinions*np.roll(self.opinions,-1,axis=1)
        con[:,:,3] =  self.opinions*np.roll(self.opinions,1,axis=1)
        boolean = (con==-1)
        interface = np.float16(boolean)
        self.DOI[t] = np.sum(interface)/(4.0*self.length*self.width)    
        
    
    
    def plot_grid(self):
        fig, ax = plt.subplots()
    
        data = self.opinions
    
        cax = ax.imshow(data, interpolation='nearest', cmap='Greys')
        ax.set_title('Opinions')
        cbar = fig.colorbar(cax, ticks=[-1, 1])
        plt.show()
        
        
    def plot_DOI(self):
        figure(3)
    
        data = self.DOI[1:self.DOI_size]
        plot((1.0/(self.length*self.width))*np.linspace(1,self.DOI_size-1,self.DOI_size-1),data)
        #plt.plot([1,2,3],[4,5,6])
        plt.xlabel('Time')
        plt.ylabel('Segregation Coefficient')
        plt.show()
        
    
    def plot_happiness(self):
        fig, ax = plt.subplots()
    
        data = self.mag
    
        cax = ax.imshow(data, interpolation='nearest')
        ax.set_title('Happiness')
        cbar = fig.colorbar(cax, ticks=[0,self.threshold, 1])
        plt.show()
        
    def plot_value(self):
        fig, ax = plt.subplots()
    
        data = self.worth
    
        cax = ax.imshow(data, interpolation='nearest')
        ax.set_title('Value')
        cbar = fig.colorbar(cax, ticks=[0,0.2,0.4,0.6,0.8, 1])
        plt.show()
        
        
        
def create_coords_list(A):
    #new_A =  [[row[i] for row in A] for i in range(np.size(A,axis=1))] #Slower????
    new_A = []
    for i in range(np.size(A,axis=1)):
        new_A.append([A[0,i],A[1,i]])
    
    return new_A    
        
def get_intersection(A,B):
    C =  []
    for i in range(int(np.size(A,axis=1))):
        A_i = A[0,i]
        A_j = A[1,i]
        for j in range(int(np.size(B,axis=1))):
            if A_i==B[0,j] and A_j == B[1,j]:
                C.append([A_i,A_j])
        
    return C

def get3_intersection(A,B,C):
    D =  []
    #print('abc',A,B,C)
    for i in range(int(np.size(A,axis=1))):
        A_i = A[0,i]
        A_j = A[1,i]
        for j in range(int(np.size(B,axis=1))):
            B_i = B[0,j]
            B_j = B[1,j]
            for k in range(int(np.size(C,axis=1))):               
                if A_i==B[0,j] and A_j == B[1,j] and A_i==C[0,k] and A_j==C[1,k]:
                    D.append([A_i,A_j])
        
    #print(D)
    return D        

def execute(l,w,dov,thr):
    a_grid = grid(l,w,dov,thr)
    t = 0
    time = 0
    
    while(np.sum(a_grid.happiness)<a_grid.n_agents):
        if(time>10.0):
            a_grid.plot_grid()
        t += 1
        time+=1.0/(length*width)
        a_grid.calculate_global_happiness()
        a_grid.calculate_global_worth()
        a_grid.update_site()
        a_grid.calculate_DOI(t)
        if a_grid.moves_avaliable_check==0:
            break
    a_grid.DOI_size = t
    #print(t,a_grid.DOI_size)
    #a_grid.plot_DOI()
    final_DOI = a_grid.DOI[t]
    Dens_Hap = np.sum(a_grid.happiness*np.abs(a_grid.opinions))/a_grid.n_agents
    time2Finish = time
    return [Dens_Hap, time2Finish, final_DOI]

n_avs = 50
n_points = 50
D_Happiness = np.zeros((n_points ,n_avs))
t_finish    = np.zeros((n_points ,n_avs))
DOI_finish    = np.zeros((n_points ,n_avs))

length = 32
width  = 32
DOV    = np.linspace(0.0,1.0,n_points )
#DOV = [0.1]
thresh = 0.625
for j in range(n_points ):
    for i in range(n_avs):
        D_Happiness[j,i], t_finish[j,i], DOI_finish[j,i] = execute(length,width,DOV[j],thresh)
        #print('Final Happiness: ',D_Happiness,'Time Taken: ', t_finish)
    print(j/n_points )

plt.figure(1)
plt.plot(DOV,np.mean(D_Happiness,axis=1))   
plt.xlabel('Density of Vacancies')
plt.ylabel('Final Happiness') 

plt.figure(2)
plt.plot(DOV,np.mean(t_finish,axis=1))   
plt.xlabel('Density of Vacancies')
plt.ylabel('Endtime') 

plt.figure(3)
plt.plot(DOV,np.mean(DOI_finish,axis=1))   
plt.xlabel('Density of Vacancies')
plt.ylabel('Segregation Coefficient') 


#figure(1)
#plt.hist(D_Happiness,bins=10)
#plt.xlabel('Final Happiness')
#
#figure(2)
#plt.hist(t_finish,bins=10)
#plt.xlabel('Time to Finish')
#plt.show()
'''
length = 64
width  = 64    
DOV    = 0.5
thresh = 0.5
a_grid = grid(length,width,DOV,thresh)
t = 0
time = 0
#print(a_grid.opinions)
while(np.sum(a_grid.happiness)<a_grid.n_agents):
    t += 1
    time+=1.0/(length*width)
    a_grid.calculate_global_happiness()
    a_grid.calculate_global_worth()
    a_grid.update_site()
    #print(time)
    if a_grid.moves_avaliable_check==0:
        print('There are no more avaliable moves')
        break
#    print(t)
    #a_grid.plot_grid()
#    a_grid.plot_happiness()
#    a_grid.plot_value()
#    print(np.sum(a_grid.happiness),a_grid.n_agents)
    if 1*t%100==0:
        print(time)
        a_grid.plot_grid()
        #a_grid.plot_happiness()
        #a_grid.plot_value()
        print(np.sum(a_grid.happiness),a_grid.n_agents)
#print(a_grid.happiness)
#print(a_grid.worth)
print(time)
a_grid.plot_grid()
#a_grid.plot_happiness()
#print(np.sum(a_grid.happiness),a_grid.n_agents)

'''