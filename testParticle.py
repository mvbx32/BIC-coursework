import unittest 
import random
from particle import Particle
import unittest
import numpy as np
from ANN_alone import Activation, Layer, ANN  # Remplacez par le vrai nom de votre module

from data import X_train, Y_train, X_test, Y_test
from tools import * 

Informants = randomParticleSet
AssessFitness = inv_ANN_MSE
informants_number = 2 

class TestParticle(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        Particle.AssessFitness = AssessFitness
        Particle.Informants = Informants
        Particle.informants_number = informants_number
        Particle.ANN_structure = [8, 5, 1]
        self.p = Particle()
    """
    def tearDown(self):
        # Réinitialisation des variables de classe
        Particle.ANN_structure = None
        Particle.fittest_solution = None
        Particle.best_fitness = -np.inf
    """
    def test_initialization_requires_structure(self):
        Particle.ANN_structure = None
        with self.assertRaises(AssertionError):
            Particle()

    def test_vector_length_matches_structure(self):
        expected_len = 8 * 5  + (5*1) + 5 * 1 + (1*1) # SUM_layers(inputs number + nb bias)
        self.assertEqual(len(self.p.vector), expected_len)

    def test_ann_model_created(self):
        self.assertIsInstance(self.p.ANN_model, ANN)
        self.assertEqual(self.p.ANN_model.layer_size, Particle.ANN_structure)
        ann_params = self.p.ANN_model.get_params()
        self.assertTrue((self.p.vector == ann_params).all()) # TO DO

    def test_default_fitness_values(self):
        self.assertTrue(np.isinf(self.p.fitness))
        self.assertTrue(np.isinf(self.p._best_fitness)) 

    def test_fittest_solution_initialized(self):
        self.assertIsNotNone(Particle.fittest_solution)
        np.testing.assert_array_equal(Particle.fittest_solution, self.p.vector)

    def test_eq_method(self):
        p2 = Particle()
        p2._vector = self.p.vector.copy()
        self.assertTrue(self.p == p2)
        p2._vector = np.array([999])
        self.assertFalse(self.p == p2)

    def test_fitness_setter_improves_best(self):
        self.p._best_fitness = 5
        old_best = self.p._best_fitness
        self.p.fitness = 10  # meilleure fitness
        self.assertEqual(self.p._best_fitness,10)
        self.assertTrue(np.array_equal(self.p.best_x, self.p.vector))
        self.assertEqual(Particle.best_fitness,10)

    def test_informants_setter_adds_self(self):
        p2 = Particle()
        informants = [[p2.vector.copy()],[2]]
        self.p.informants = informants

        isIn = False
        for el in self.p.informants : 
            if (el == self.p.vector).all():
                isIn = True
                break 
        self.assertTrue(isIn)
    
    def test_informants_setter_updates_best_informant(self):
 
        p1 = self.p 
        self.assertTrue( ( p1.best_informant == p1.vector).all()) # initialisation
        p2 = Particle()
        p1.fitness = 5
        p2.fitness = 10
        p1.informants = [[p2.vector.copy()],[p2.fitness]]
        self.assertGreaterEqual(p1.best_informant_fitness, 10)

    """ # TODO
   
    """
    """  def test_copy(self):
        p1 = Particle()
        p2 = p1.copy()

        print(p1,p2)
        self.assertFalse(p1.ANN_model is p2.ANN_model)  # False ✅
        self.assertTrue(np.allclose(p1.vector, p2.vector))  # True ✅

        p2.vector[0] += 1
        self.assertFalse(np.allclose(p1.vector, p2.vector))  # False ✅ """

    
    def test_AssessFitness_Func(self):
        
        # definition of the Fitness function
        a = lambda x : "Fitness function modified for test"
        Particle.AssessFitness = a
        self.assertEqual(a, Particle.AssessFitness) 

        # back to the previous one
    
        Particle.AssessFitness = AssessFitness
        self.p.assessFitness()
        self.assertEqual(AssessFitness(self.p), self.p.fitness, self.p)

    

        

    

if __name__ == '__main__':
    unittest.main()
    
