#!/usr/bin/python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import axes3d


# defining the class for later 3D arrow plots
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
  
def plot_energies():
    infile = open('Data/energy_plot_values.pkl', 'r+b')
    energy_plot_values = pickle.load(infile)
    infile.close()
    dt = energy_plot_values[0]
    vmax = energy_plot_values[1]
    grid_size = energy_plot_values[2]
    thermostat = energy_plot_values[3]
    steps = energy_plot_values[4]

    infile = open('Data/E_kin.pkl', 'r+b')
    E_kin = pickle.load(infile)
    infile.close()
    
    infile = open('Data/E_pot.pkl', 'r+b')
    E_pot = pickle.load(infile)
    infile.close()
    
    infile = open('Data/E_tot.pkl', 'r+b')
    E_tot = pickle.load(infile)
    infile.close()
    
    t = np.arange(steps)
    fig2 = plt.figure()      
    axe2 = fig2.add_subplot(111)
    axe2.set_ylabel('E')
    axe2.set_xlabel('n$_{steps}$')
    axe2.set_title('Energies for v$_{max}$ = %5.1f, n$_{steps}$= %5.0f, $\Delta$t=%5.3f' %(vmax, steps, dt))
    axe2.plot(t, E_kin, label='$E_{kin}$')
    axe2.plot(t, E_pot, label='$E_{pot}$')
    axe2.plot(t, E_tot, label='$E_{tot}')
    axe2.legend(loc=0)
    fig2.savefig('Graphs/Energies_grid'+str(grid_size)+'_vmax'+str(vmax)+'_thermo'+str(thermostat)+'.png')

def plot_RDF():
    infile = open('Data/RDF_plot_values.pkl', 'r+b')
    RDF_plot_values = pickle.load(infile)
    infile.close()
    dt = RDF_plot_values[0]
    vmax = RDF_plot_values[1]
    grid_size = RDF_plot_values[2]
    thermostat = RDF_plot_values[3]
    steps = RDF_plot_values[4]
   
    infile = open('Data/RDF.pkl', 'r+b')
    RDF = np.array(pickle.load(infile), dtype=np.float_)
    infile.close()
    
    RDF /= np.sum(RDF)
    
    infile = open('Data/bins.pkl', 'r+b')
    bins = pickle.load(infile)
    infile.close()
     
    width = 0.7*(bins[1]-bins[0])
    left = bins[:-1]
     
    fig3 = plt.figure()
    axe3 = fig3.add_subplot(111)
    axe3.bar(left, RDF, width=width)
    axe3.set_title('RDF for v$_{max}$ = %5.1f, n$_{steps}$= %5.0f, $\Delta$t=%5.3f, thermostat: %r' %(vmax, steps, dt, thermostat))
    axe3.set_xlim(xmax = float(grid_size)/2)
    axe3.set_xlabel('Distance r')
    axe3.set_ylabel('Propability P')
    fig3.savefig('Graphs/RDF_grid'+str(grid_size)+'_vmax'+str(vmax)+'_thermo'+str(thermostat)+'.png')


# plot_positions plots all positions in a 3D plot and saves it as a png
def plot_positions():
  infile = open('Data/RDF_plot_values.pkl', 'r+b')
  RDF_plot_values = pickle.load(infile)
  infile.close()
  dt = RDF_plot_values[0]
  vmax = RDF_plot_values[1]
  grid_size = RDF_plot_values[2]
  thermostat = RDF_plot_values[3]
  steps = RDF_plot_values[4]
  
  for n in range(int(steps)):
    infile = open('Data/x'+str(n)+'.pkl', 'r+b')
    x = np.array(pickle.load(infile))
    infile.close()

    plt.ioff()
    fig = plt.figure()
    axe = fig.add_subplot(111, projection='3d')
    xs, ys, zs = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    for i,el in enumerate(x):
      xs[i] = el[0]
      ys[i] = el[1]
      zs[i] = el[2]
    axe.scatter(xs,ys,zs, s=500)
    ''' 
    for i, acc in enumerate(a):
      acc /= 10
      acc += x[i]
      axe.add_artist(Arrow3D(*zip(x[i],acc), mutation_scale=20, lw=1, arrowstyle="-|>", color="k"))
    '''
    axe.set_xlim(0, grid_size)
    axe.set_ylim(0, grid_size)
    axe.set_zlim(0, grid_size)
    axe.set_title('Step %5.0f' %n)
    fig.savefig('Movie/step'+str(n)+'.png')
    plt.close('all')

