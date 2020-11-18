#!/usr/bin/env python
from schnetpack.md.simulation_hooks.barostats import BarostatHook
from schnetpack.md.utils import MDUnits
import torch
import sys

class MonteCarloBarostat(BarostatHook):
    """
    Monte Carlo Barostat class for regulating the pressure
    with SchNetPack MD simulations. Based off of OpenMM's
    MC Barostat at https://github.com/openmm/openmm/blob/master/openmmapi/src/MonteCarloBarostatImpl.cpp
    Uses BarostatHook as the base class
    """
    def __init__(self, target_pressure, temperature_bath, frequency=25, detach=True):

        super(MonteCarloBarostat, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            detach=detach
        )
        #Convert pressure to Ha/nm^3
        self.target_pressure = self.target_pressure * MDUnits.unit2unit("kj/mol", "Ha")
        #Frequency to attempt MC move
        self.frequency = frequency
        #Convert kT to Ha
        self.kb_temperature = self.temperature_bath * MDUnits.kB * MDUnits.unit2unit("kj/mol", "Ha")

    def _init_barostat(self, simulator):
        #Initial amount to scale volume
        self.volScale = 0.01*simulator.system.volume
        self.numAttempted = 0
        self.numAccepted = 0
        self.step = 0

        #Initialize on cuda with random seed
        torch.cuda.seed()

    def _apply_barostat(self, simulator):
        self.step += 1
        #Do nothing if the step is less than the MC move frequency
        if self.step < self.frequency:
            pass
        else:
            self.step = 0
            #Get initial energy of the system
            initial_energy = simulator.system.properties["energy"]
            #Get volume
            volume = simulator.system.volume
            #Get random number to increase/decrease volume by
            deltaVol = self.volScale*2.0*(torch.rand(1, device='cuda') - 0.5)
            newVol = volume + deltaVol
            #Determines how much to scale each side of the box by
            lengthScale = torch.pow(newVol/volume, 1.0/3.0)
            #Get initial posiitons, cell vectors and forces. Will be used to
            #revert system back if move is not accepted
            initial_pos = simulator.system.positions.clone()
            initial_cell = simulator.system.cells.clone()
            initial_forces = simulator.system.forces.clone()
            #Scale the coordinates by lengthScale
            self._scale_coordinates(simulator, lengthScale, lengthScale, lengthScale)
            #Scale box vectors by lengthScale
            simulator.system.cells *= lengthScale

            #Calculate system to get new energy
            simulator.calculator.calculate(simulator.system)
            final_energy = simulator.system.properties["energy"]
            #Convert energy change from eV (model units) to hartree
            dE = (final_energy - initial_energy) * MDUnits.unit2unit("ev", "Ha")

            #Units are Ha + Ha/nm^3 * nm^3 - Ha 
            w = dE + self.target_pressure * deltaVol - self.kb_temperature * torch.log(newVol/volume)
            #Determine whether to accept or reject MC move
            if w > 0 and torch.rand(1, device='cuda') > torch.exp(-w/self.kb_temperature):
                #Step rejected, revert to initial system
                simulator.system.positions = initial_pos
                simulator.system.cells = initial_cell
                simulator.system.forces = initial_forces
                volume = newVol
            else:
                #Step accepted
                self.numAccepted += 1

            self.numAttempted += 1
            if self.numAttempted >= 10:
                if self.numAccepted < 0.25*self.numAttempted:
                    self.volScale /= 1.1
                    self.numAttempted = 0
                    self.numAccepted = 0
                elif self.numAccepted > 0.75 * self.numAttempted:
                    self.volScale = torch.min(self.volScale*1.1, volume*0.3)
                    self.numAttempted = 0
                    self.numAccepted = 0

    def _scale_coordinates(self, simulator, scX, scY, scZ):
        #Scale com by scX, scY and scZ
        dr = torch.sub(simulator.system.center_of_mass, simulator.system.positions)
        scale = torch.tensor([[[scX, scY, scZ]]]).cuda()
        com = torch.mul(simulator.system.center_of_mass, scale)
        simulator.system.positions = torch.sub(com, dr)

    def on_step_end(self, simulator):
        #Don't do anything on step end unlike other barostats
        pass

class MonteCarloAnisotropicBarostat(MonteCarloBarostat):
    """
    Monte Carlo Barostat class for regulating the pressure
    with SchNetPack MD simulations. This allows for altering
    each side of the periodic box individually, which permits changes in shape and size
    Based off of OpenMM's MC AnisotropicBarostat at 
    https://github.com/openmm/openmm/blob/master/openmmapi/src/MonteCarloAnisotropicBarostatImpl.cpp
    Uses MonteCarloBarostat as the base class
    """
    def __init__(self, target_pressure, temperature_bath, frequency=25, detach=True):

        super(MonteCarloAnisotropicBarostat, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            detach=detach
        )

    def _init_barostat(self, simulator):
        #volScale will now be a size 3 tensor, as each side of the box can be scaled individually
        self.volScale = 0.01*torch.repeat_interleave(simulator.system.volume, 3, dim=0).reshape(3)
        #Keep track of moves attempted and accepted for each side of the box
        self.numAttempted = torch.zeros(3)
        self.numAccepted = torch.zeros(3)
        self.step = 0
        torch.cuda.seed()

    def _apply_barostat(self, simulator):
        self.step += 1
        #Do nothing if the step is less than the MC move frequency
        if self.step < self.frequency:
            pass
        else:
            self.step = 0
            #Get initial energy of the system
            initial_energy = simulator.system.properties["energy"]

            #Randomly select box side for move application
            rnd = torch.rand(1, device='cuda') * 3.0
            if rnd < 1.0: ax = 0
            elif rnd < 2.0: ax = 1
            else: ax = 2

            volume = simulator.system.volume
            #Get amount to change volume in terms of one box side
            deltaVol = self.volScale[ax]*2.0*(torch.rand(1, device='cuda') - 0.5)
            newVol = volume + deltaVol
            lengthScale = torch.ones(3)
            #Alter legthScale for one box side
            lengthScale[ax] = newVol/volume
            #Get initial position, cell and forces
            initial_pos = simulator.system.positions.clone()
            initial_cell = simulator.system.cells.clone()
            initial_forces = simulator.system.forces.clone()

            #Scale coordinates and cell
            self._scale_coordinates(simulator, lengthScale[0], lengthScale[1], lengthScale[2])
            simulator.system.cells[0, 0, 0, :] *= lengthScale[0]
            simulator.system.cells[0, 0, 1, :] *= lengthScale[1]
            simulator.system.cells[0, 0, 2, :] *= lengthScale[2]

            #Get new energy, this part is the same as in MC barostat
            simulator.calculator.calculate(simulator.system)
            final_energy = simulator.system.properties["energy"]
            dE = (final_energy - initial_energy) * MDUnits.unit2unit("ev", "Ha")
            w = dE + self.target_pressure * deltaVol - self.kb_temperature * torch.log(newVol/volume)

            if w > 0 and torch.rand(1, device='cuda') > torch.exp(-w/self.kb_temperature):
                simulator.system.positions = initial_pos
                simulator.system.cells = initial_cell
                simulator.system.forces = initial_forces
                volume = newVol
            else:
                self.numAccepted[ax] += 1

            self.numAttempted[ax] += 1
            if self.numAttempted[ax] > 10:
                if self.numAccepted[ax] < 0.25 * self.numAttempted[ax]:
                    self.volScale[ax] /= 1.1
                    self.numAttempted[ax] = 0
                    self.numAccepted[ax] = 0
                elif self.numAccepted[ax] > 0.75 *self.numAttempted[ax]:
                    self.volScale[ax] = torch.min(self.volScale[ax]*1.1, volume*0.3)
                    self.numAttempted[ax] = 0
                    self.numAccepted[ax] = 0

