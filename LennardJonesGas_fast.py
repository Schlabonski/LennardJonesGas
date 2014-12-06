#!/usr/bin/python3

# This is the fast version of the simulation, optimized to be interpreted with PyPy.
# To get plots from the data, run plotting.py with a python version that supports
# matplotlib.pyplot.

import numpy as np
import sys
import itertools as it
import pickle

# DistancePBC delivers the smallest distance between two points according to the periodix boundary conditions PBC
def DistancePBC(x,y,grid_size, vectorial=False):
  delta = x-y
  delta = np.where((np.abs(delta) > 0.5*grid_size), np.where(delta<0, -delta - grid_size, grid_size-delta), -delta)
  if vectorial==True:
    return delta, np.sqrt(np.ndarray.sum(delta**2))
  else:
    return np.sqrt(np.ndarray.sum(delta**2))

#LennardJones returns potential energy for each particle of a pair of particles
def LennardJones(x_i, x_j, grid_size):
  r_ij = DistancePBC(x_i, x_j, grid_size)
  V = 1./r_ij**12 - 1./r_ij**6
  return V

#LennardJonesForce returns the force acting on a particle on position x_i from a particle on x_j(vectors)
def LennardJonesForce(x_i, x_j, grid_size):
  r_vec, r_ij = DistancePBC(x_i, x_j, grid_size, vectorial=True)
  F = r_vec/r_ij * (6*r_ij**(-7) - 12*r_ij**(-13))
  return F, r_vec

# TotalForceEfficient returns the sum of all forces acting on the particle x_i, x should be the array containing all particle positions, the globally defined variables Forces and relative_positions are necessary to calculate the virial pressure
def TotalForceEfficient(x, grid_size, initiate=False):
  global Forces
  Forces = np.zeros((len(x), len(x), 3))
  global relative_positions
  relative_positions = np.zeros((len(x), len(x), 3))
  for i,j in it.combinations(range(len(x)), 2):
    Forces[i,j], relative_positions[i,j] = LennardJonesForce(x[i], x[j], grid_size)
    Forces[j,i] = (-1)*Forces[i,j]
    relative_positions[j,i] = (-1)*relative_positions[i,j]

  if initiate == True:
    return
  return np.ndarray.sum(Forces, axis=1)
  

# TotalPotentialEnergyEfficient returns the total potential energy of a position configuration x (array of position-vectors)
def TotalPotentialEnergyEfficient(x, grid_size):
  E = np.zeros((len(x),len(x)))
  for i,j in it.combinations(range(len(x)),2):
    E[i,j] = E[j,i] = LennardJones(x[i], x[j], grid_size)
  return np.ndarray.sum(E)

# TotalKineticEnergy returns the total kinetic energy of an array of velocity vectors
def TotalKineticEnergy(v):
  return np.ndarray.sum(0.5*v**2)

# VerletVelocity returns the x(n+1) and v(n+1), when given x,v and a as arrays
def VerletVelocity(x, v, dt, grid_size):
  a = TotalForceEfficient(x, grid_size)
  x_next = x + v*dt + (a/2)*dt**2
  a_next = TotalForceEfficient(x_next, grid_size)
  v_next = v + (a_next+a)/2*dt
  return np.mod(x_next, grid_size), v_next

# BerendsenThermostat returns a rescaling factor for the velocities v of the particles to keep the systems temperature around T_0, tau is a coupling constant, dt is the integration timestep of the system
def BerendsenThermostat(T, T_0, tau, dt):
  lambda_rescaling = np.sqrt(1 + (float(dt)/tau)*((float(T_0)/T) - 1))
  return lambda_rescaling 

# BerendsenBarostat returns a rescaling factor mu based on the berendsen theory
def BerendsenBarostat(F, r, E_kin, P_0, grid_size, tau_p, dt, beta=1, initiate=False):
  Virial = np.sum(np.triu(F)*np.triu(r))
  if initiate == True:
    return 1./(3*grid_size**3)*(E_kin-Virial)
  P = 1./(3*grid_size**3)*(E_kin-Virial)
  mu = (1 + float(beta*dt)/tau_p*(P-P_0))**(1./3)
  
  return mu

#RDFHistogramEfficient returns an array of all occuring distances from a set of position vectors x
def RDFHistogramEfficient(x, grid_size):
  def dist(j):
    return DistancePBC(x[j[0]], x[j[1]], grid_size)
  #r = [dist(j) for j in it.combinations(range(len(x)),2)]
  r = np.array(list(map(dist, it.combinations(range(len(x)),2))))
  return np.histogram(r, bins=100, range=(0.5, float(grid_size)/2))

# initiate_E_fixed creates random positions and velocities of particles in a grid sized gridsize^3 for a fixed total energy E_tot
def initiate_E_fixed(grid_size, N_part, E_tot):
  grid_positions = np.zeros((grid_size**3,3))
  v = np.zeros((N_part,3))
  x = np.zeros((N_part,3))

  # define all possible positions in grid
  n=0
  for i in range(grid_size):
    for j in range(grid_size):
      for k in range(grid_size):
        grid_positions[n] = np.array([i,j,k])
        n += 1

  # in this case all positions in the grid are occupied
  if N_part == grid_size**3:
    x = grid_positions

  # else the occupied positions in the grid will be selected randomly
  elif N_part < grid_size**3:
    for i in range(N_part):
      rand = np.random.random_integers(len(grid_positions))
      x[i] = grid_positions[rand]
      grid_positions = np.delete(grid_positions, rand, axis = 0)

  # now the potential energy of the configuration will be calculated
  E_pot = TotalPotentialEnergyEfficient(x, grid_size)

  # the remaining energy will be equally distributed as initial velocity of the particles
  E_kin = E_tot - E_pot
  v_init = np.sqrt(E_kin)
  for n in range(N_part):
    theta = np.random.random()*2*np.pi
    phi = np.random.random()*np.pi
    v[n] = v_init*np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])

  return x,v

# initiate_vmax_fixed returns random positions in a grid sized grid_size^3, the initial velocities are chosen randomly from the interval [0, vmax)
def initiate_vmax_fixed(grid_size, N_part, vmax = 1):
  grid_positions = np.zeros(((grid_size)**3,3))
  v = np.zeros((N_part,3))
  x = np.zeros((N_part,3))

  # define all possible positions in grid
  n=0
  for i in range(grid_size):
    for j in range(grid_size):
      for k in range(grid_size):
        # the position +0.5 is chosen, so that initially there is no particle directly next to the border
        grid_positions[n] = np.array([i+0.5,j+0.5,k+0.5])
        n += 1

  # in this case all positions in the grid are occupied
  if N_part == (grid_size)**3:
    x = grid_positions

  # else the occupied positions in the grid will be selected randomly
  elif N_part < (grid_size)**3:
    for i in range(N_part):
      rand = np.random.random_integers(len(grid_positions)-1)
      x[i] = grid_positions[rand]
      grid_positions = np.delete(grid_positions, rand, axis = 0)

  # the initial velocities will be rnd([0,vmax)), with random directions
  for n in range(N_part):
    theta = np.random.random()*2*np.pi
    phi = np.random.random()*np.pi
    v_init = np.random.random()*vmax
    v[n] = v_init*np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])

  return x, v

def Simulation(grid_size, N_part, T=0.5, steps=100, vmax=2, plot_movement=True, plot_energies=True, plot_RDF=True, thermostat=True, barostat=True):
  dt = float(T)/steps
  x, v = initiate_vmax_fixed(grid_size, N_part, vmax)
  E_kin_init = TotalKineticEnergy(v)
  E_kin = []
  E_pot = []
  E_tot = []
  Temperatures = []
  positions = []

  if thermostat == True:
    T_0 = np.ndarray.sum(v**2)
    tau = 10*dt

  if barostat == True:
    tau_p = 10*tau
    TotalForceEfficient(x, grid_size, initiate=True)
    P_0 = BerendsenBarostat(Forces, relative_positions, E_kin_init,0, grid_size, tau_p, dt, initiate=True)
    
  if plot_RDF == True:
    RDF, bins = RDFHistogramEfficient(x, grid_size)

  for n in range(steps):
    E_kin_i = TotalKineticEnergy(v)
    E_pot_i = TotalPotentialEnergyEfficient(x, grid_size)
    E_tot_i = E_kin_i + E_pot_i
    T_i = np.ndarray.sum(v**2)
    
    E_pot.append(E_pot_i)
    E_kin.append(E_kin_i)
    E_tot.append(E_tot_i)
    Temperatures.append(T_i)
            
    if thermostat == True:
      v *= BerendsenThermostat(T_i, T_0, tau, dt)

    if plot_movement == True:
      outfile = open('Data/x'+str(n)+'.pkl', 'w+b')
      pickle.dump(x.tolist(), outfile)
      outfile.close()
      #acc = (-1)*TotalForceEfficient(x, grid_size)
      #np.save('Data/acc', acc)
    
    if plot_RDF ==True:
      RDF_i = RDFHistogramEfficient(x, grid_size)[0]
      RDF += RDF_i
             
    x, v = VerletVelocity(x, v, dt, grid_size)
    
    if barostat == True:
      mu = BerendsenBarostat(Forces, relative_positions, E_kin_i, P_0, grid_size, tau_p, dt)
      grid_size *= mu
      x *= mu

    print '{0}\r'.format(str(int(float(n)/steps*100)) + '% done.'+ '\t'+'E_tot = %5.3f' %E_tot_i)
    #print 'E_tot = %5.3f' %E_tot_i + '\r',
    
  if plot_energies == True:
    outfile = open('Data/energy_plot_values.pkl', 'w+b')
    energy_plot_values = np.array([dt, vmax, grid_size, int(thermostat), steps])
    pickle.dump(energy_plot_values.tolist(), outfile)
    outfile.close()

    outfile = open('Data/E_kin.pkl', 'w+b')
    E_kin = np.array(E_kin)
    pickle.dump(E_kin.tolist(), outfile)
    outfile.close()

    outfile = open('Data/E_pot.pkl', 'w+b')
    E_pot = np.array(E_pot)
    pickle.dump(E_pot.tolist(), outfile)
    outfile.close()
    
    outfile = open('Data/E_tot.pkl', 'w+b')
    E_tot = np.array(E_tot)
    pickle.dump(E_tot.tolist(), outfile)
    outfile.close()
    

  if plot_RDF ==True:
    outfile = open('Data/RDF_plot_values.pkl', 'w+b')
    RDF_plot_values = np.array([dt, vmax, grid_size, int(thermostat), steps])
    pickle.dump(RDF_plot_values.tolist(), outfile)
    outfile.close()
    
    outfile = open('Data/RDF.pkl', 'w+b')
    pickle.dump(RDF.tolist(), outfile)
    outfile.close()

    outfile = open('Data/bins.pkl', 'w+b')
    pickle.dump(bins.tolist(), outfile)
    outfile.close()

Simulation(6, 6**3, T=1, steps=1000, vmax = 0.05, thermostat=True)
