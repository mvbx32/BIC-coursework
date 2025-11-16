import sys
sys.path.insert(0, './')

import unittest 
import random
from particle import Particle
import unittest
import numpy as np
from ANN_alone import Activation, Layer, ANN  # Remplacez par le vrai nom de votre module

from pso import PSO
from data import Data
from tools import * 

Informants = randomParticleSet
AssessFitness = inv_ANN_MAE
informants_number = 2 

ANNStructure = [8,5,1]
ANN_activation = "linear"
swarmsize = 10 

alpha = 1 
beta  = 1 
gamma = 1
delta = 1 

epsi  = 0.3  
informants_number = 3
max_iteration_number = 1

class TestPSO(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        
        pso = PSO(swarmsize, 
        alpha, 
        beta, 
        gamma,
        delta,
        epsi, 
        ANNStructure, 
        ANN_activation,
        AssessFitness, 
        informants_number, 
        Informants, 
        max_iteration_number = max_iteration_number, 
        verbose = -1) 


        
    def variables_initialisation(self):
        self.assertEqual(self.pso.P,[])
        self.assertIsNone(self.pso.Best)
        self.assertIsNone(self.pso.BestANN)
        self.assertTrue(self.pso.BestFitness == -1*np.inf)
        
    def test_No_informants_component_if_NULL(self):
        pass
        
    """ #?? HOW TO TEST
    def test_fittest_solution_initialized(self):
        self.assertIsNotNone(Particle.fittest_solution)
        np.testing.assert_array_equal(Particle.fittest_solution, self.p.vector) """ #PSO

    """
    def test_fitness_setter_improves_best(self):

        self.p._best_fitness = 5
        old_best = self.p._best_fitness
        self.p.fitness = 10  # meilleure fitness
        self.assertEqual(self.p._best_fitness,10)
        self.assertTrue(np.array_equal(self.p.best_x, self.p.vector))
        self.assertEqual(Particle.best_fitness,10)"""
    

if __name__ == '__main__':
    unittest.main()
    
