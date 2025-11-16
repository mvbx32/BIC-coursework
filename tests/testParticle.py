import sys
sys.path.insert(0, './')

import unittest 
import random
from particle import Particle
import unittest
import numpy as np
from ANN_alone import Activation, Layer, ANN  # Remplacez par le vrai nom de votre module

from data import Data
from tools import * 

Informants = randomParticleSet
AssessFitness = inv_ANN_MAE
informants_number = 2 


class TestParticle(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        self.AssessFitness = AssessFitness
        self.Informants = Informants
        self.informants_number = informants_number
        self.ANN_activation = "linear"
        self.ANN_structure = [8,'input', 5,'relu', 1,'relu']

        m = Particle.particleNumber 
        self.p = Particle(self.ANN_structure)

        self.assertEqual(m+1,self.p.id)
    
    """

    def test_initialization_requires_structure(self):
        self.ANN_structure = None
        with self.assertRaises(AssertionError):
            p = Particle()
            
    """

    # ANN and vector representation
    def test_vector_length_matches_structure(self):
        
        ANN_layers = [ self.ANN_structure[i]  for i in range(len(self.ANN_structure)) if i%2==0 ]# 8 * 5  + (5*1) + 5 * 1 + (1*1) # SUM_layers(inputs number + nb bias)

        expected_len = np.sum([ (ANN_layers[i-1] + 1)*ANN_layers[i]  for i in range(1,len(ANN_layers))  ])# 8 * 5  + (5*1) + 5 * 1 + (1*1) # SUM_layers(inputs number + nb bias)
        self.assertEqual(len(self.p.vector), expected_len)


    def test_ann_model_created(self):
        self.assertIsInstance(self.p.ANN_model, ANN)
        self.assertEqual(self.p.ANN_model.layer_size, [self.ANN_structure[i] for i in range(len(self.ANN_structure)) if i%2 ==0  ] ) 
        ann_params = self.p.ANN_model.get_params()
        self.assertTrue((self.p.vector == ann_params).all()) 

    def test_default_fitness_values(self):
        self.assertTrue(np.isinf(self.p.fitness))
        self.assertTrue(np.isinf(self.p._best_fitness)) 

    def test_eq_method(self):
        p2 = Particle(self.ANN_structure)
        p2._vector = self.p.vector.copy()
        self.assertTrue(self.p == p2)
        p2._vector = np.array([999])
        self.assertFalse(self.p == p2)

    def test_copy(self):
        p1 = Particle(self.ANN_structure)
        p2 = p1.copy()

        self.assertFalse(p1.ANN_model is p2.ANN_model)  # False 
        self.assertTrue(np.allclose(p1.vector, p2.vector))  # True

        p2.vector[0] += 1
        self.assertFalse(np.allclose(p1.vector, p2.vector))  # False  """

    def test_vector_setter(self):

        ANN_layers = [ self.ANN_structure[i] for i in range(len(self.ANN_structure)) if i%2 == 0]
        ANN_acti = [ self.ANN_structure[i] for i in range(len(self.ANN_structure)) if i%2 == 1]
        a2 = ANN(ANN_layers, ANN_acti)
        self.p.vector = a2.get_params()

        self.assertTrue((self.p._vector == self.p.vector).all()  # Does the accesseur modify well the variable ?
                        and (a2.get_params() == self.p._vector).all() ) # Is the value updated ?
        self.assertTrue(self.p.ANN_model == a2)

    def test_fitness_setter(self): 

        p1 = Particle(self.ANN_structure)
        p1.assessFitness(self.AssessFitness)
        p2 = Particle(self.ANN_structure)
        p2._fitness = p1.fitness + 10
    
        p1.vector = p2.vector # as if we were updating p value at the end of the PSO
        p1.fitness = p2.fitness 

        self.assertTrue( (p1.best_x == p2.vector).all())

        p1 = Particle(self.ANN_structure)
        p1.assessFitness(self.AssessFitness)
        p2 = Particle(self.ANN_structure)
        p2._fitness = p1.fitness - 10
        p1.vector = p2.vector # as if we were updating p value at the end of the PSO
        p1.fitness = p2.fitness 
        self.assertFalse( (p1.best_x == p2.vector).all())

    def test_informants_setter_adds_self(self):
        # PSO l18 : x has to belongs to the informants
        
        def sub_test1(p1,p2):
            # == if informant NotEmpty == 
            #Set an informant list without self.p1
            self.informants_number = 1
            informants = ([p2.vector.copy()],[2]) # (informants vector list, fitness list)
            p1.informants = informants

            #Check if p is in the informants
            isIn = False
            for el in p1.informants : 
                if (el == p1.vector).all():
                    isIn = True
                    break 
            self.assertTrue(isIn)

            self.assertEqual(self.informants_number, len(p1.informants))

            # == if informants Empty == 

            p3 = self.p.copy()

            #Set an informant list without self.p
            self.informants_number = 0
            informants = ([],[]) # (informants vector list, fitness list)
            p3.informants = informants
            self.assertEqual(self.informants_number, len(p3.informants))
        
        # Case 1 #  p1 Not in the generated informants
        p1 = self.p.copy()

        #Set a p2 particle such that p2 != p1
        p2 = self.p.copy()
        p2.vector[0] -= 1 
        self.assertNotEqual(p2,p1)
        sub_test1(p1,p2)

        # Case 2 #  p1 is in the generated informants
        sub_test1(p1,p1)

    def test_informants_setter_updates_best_informant(self):# NOK
        
        def sub_test(p2):
            
            p1 = self.p.copy()
            p1.assessFitness(self.AssessFitness)
            self.assertTrue( ( p1.best_informant == p1.vector).all()) # initialisation ??? WHAT IF NO INFORMANTS
            p1.fitness = 5 #NOK !!! depends on the order of the definition of the fitnesses

            p1.informants = [[p2.vector.copy()],[p2.fitness]]
            
            return (p1.best_informant_fitness >= p2.fitness), p1.best_informant, p1.best_informant_fitness

        p2 = Particle(self.ANN_structure)
        p2.fitness = 10

        greater,infor, fit = sub_test(p2)

        self.assertTrue(greater and (infor == p2.vector).all() and fit == p2.fitness)
        p2.fitness = -3
        greater,infor, fit = sub_test(p2)
        self.assertTrue(greater and (infor != p2.vector).all() and fit == 5)
        
        #What if EMPTY ? 
        p1 = self.p.copy()
        p1.assessFitness(self.AssessFitness)
        self.assertTrue( ( p1.best_informant == p1.vector).all()) # initialisation ??? WHAT IF NO INFORMANTS
        p1.fitness = 5 #NOK !!! depends on the order of the definition of the fitnesses
        p1.informants = [[],[]]
        self.assertTrue( ( p1.best_informant == p1.vector).all()) # initialisation ??? WHAT IF NO INFORMANTS

    def test_AssessFitness_Func(self):
        
        # back to the previous one
        self.p.assessFitness(self.AssessFitness)
        self.assertEqual(AssessFitness(self.p), self.p.fitness, self.p)

    """def test_export(self):

        export = str(self.p.vector)
        with open("test","w") as f : 
            f.write(export)
        f.close()
        print(export)
        pass"""
    
    

if __name__ == '__main__':
    unittest.main()
    
