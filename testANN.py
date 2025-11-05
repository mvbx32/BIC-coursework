import unittest 
import random
from particle import Particle
import unittest
import numpy as np
from ANN_alone import Activation, Layer, ANN  # Remplacez par le vrai nom de votre module
from pso import *

from data import X_train, Y_train, X_test, Y_test
from tools import * 
# %% Example 1 

Informants = randomParticleSet
AssessFitness = inv_ANN_MSE
#WARNING : TAKE CARE OF THE FACT THAT THE TEST INSTANTIATION HAVE THE SAME DIMENSIONS THAT THE ONE USED IN THE PSO otherwise error using the data set


Particle.AssessFitness = AssessFitness
Particle.Informants = Informants
class TestActivation(unittest.TestCase):

    def test_sigmoid(self):
        x = np.array([0, 2])
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_almost_equal(Activation.sigmoid(x), expected)

    def test_relu(self):
        x = np.array([-1, 0, 3])
        expected = np.array([0, 0, 3])
        np.testing.assert_array_equal(Activation.relu(x), expected)

    def test_tanh(self):
        x = np.array([-1, 0, 1])
        expected = np.tanh(x)
        np.testing.assert_almost_equal(Activation.tanh(x), expected)

    def test_linear(self):
        x = np.array([-1, 0, 1])
        expected = x
        np.testing.assert_almost_equal(Activation.linear(x), expected)

class TestLayer(unittest.TestCase):

    def test_layer_initialization(self):
        n_input, n_output = 4, 3
        layer = Layer(n_input, n_output, activation="relu")

        # Vérifie la taille des poids et des biais
        self.assertEqual(layer.W.shape, (n_input, n_output))
        self.assertEqual(layer.b.shape, (n_output,))
        self.assertTrue(callable(layer.activation))

    def test_layer_activation_type(self):
        layer = Layer(2, 2, activation="sigmoid")
        self.assertEqual(layer.activation, Activation.sigmoid)

    def test_equal(self):
      
        layer1 = Layer(2, 2, activation="sigmoid")
        layer2 = Layer(2,2, activation="sigmoid")
        layer2.W = layer1.W.copy()
        layer2.b = layer1.b.copy()

        self.assertTrue(layer1 == layer2)
        layer1.W[0,0]-= 100 
        self.assertNotEqual(layer1.W[0,0] ,  layer2.W[0,0] )
        self.assertNotEqual(layer1, layer2)
    def test_copy(self):
        layer1 = Layer(4, 2, activation="sigmoid")
        layer2 = layer1.copy()

        self.assertTrue(layer1 == layer2)
        layer1.W[0,0]-= 100 
        self.assertNotEqual(layer1.W[0,0] ,  layer2.W[0,0] )
        self.assertNotEqual(layer1, layer2)


class TestANN(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)  # Pour rendre les tests reproductibles
        self.ann = ANN(layer_sizes=[3, 5, 2], activations=["relu", "tanh"])

    def test_layer_sizes_property(self):
        expected_sizes = [3, 5, 2]
        self.assertEqual(self.ann.layer_size, expected_sizes)

    def test_forward_output_shape(self):
        X = np.random.rand(4, 3)  # 4 échantillons, 3 features
        Y = self.ann.forward(X)
        self.assertEqual(Y.shape, (4, 2))

    def test_get_params_length(self):
        #TODO perfectible

        ann2 = ANN([2,2,1],'relu')
        ann2.layers[1].W = np.array([[1,2],
                                     [3,4]])
        ann2.layers[2].W = np.array([[1,2],
                                     [3,4]])
        ann2.layers[1].b = np.array([[1],[2]])
        ann2.layers[2].b = np.array([[1],[2]])
        
        expected_params = np.array([1,2,3,4,1,2])

        self.assertEqual(expected_params,ann2.get_params())
        params = self.ann.get_params()
        total_expected = sum(l.W.size + l.b.size for l in self.ann.layers)
        self.assertEqual(len(params), total_expected)

    def test_set_params_restores_values(self):
        params_original = self.ann.get_params().copy()

        # Modifier les paramètres aléatoirement
        new_params = np.random.randn(len(params_original))
        self.ann.set_params(new_params)

        # Vérifie que les valeurs ont bien changé
        np.testing.assert_array_almost_equal(self.ann.get_params(), new_params)

        # Restaurer les valeurs originales
        self.ann.set_params(params_original)
        np.testing.assert_array_almost_equal(self.ann.get_params(), params_original)

    def test_invalid_activation_count(self):
        with self.assertRaises(AssertionError):
            ANN(layer_sizes=[3, 4, 2], activations=["relu"])  # manque une activation

    def test_ANN_eq__(self):

        ann2 = ANN(layer_sizes=self.ann.layer_sizes, activations=self.ann.activations)
        i = 0
        for layer in self.ann.layers : 
            ann2.layers[i] = layer.copy()
            i+=1

        ann2.set_params(self.ann.get_params())
        self.assertTrue(ann2 == self.ann)
        ann3 = ANN(layer_sizes=self.ann.layer_sizes, activations=self.ann.activations)
        self.assertNotEqual(ann3, self.ann)
        pass

    def test_copy(self):

        ann2 = self.ann.copy()
        self.assertEqual(ann2, self.ann)
        new_layer = self.ann.layers[0].copy() 
        new_layer.W[0,0] +=100
        self.ann.layers[0] = new_layer
        self.assertNotEqual(ann2, self.ann)

        pass
        

#%%


""" class Test_particle(unittest.TestCase) : 

    def __init__(self):
        pass 
        
     Is there a way to instantiate an object example that will be reset at each test ? 
        
    def test_fitness(self):
        pass """


if __name__ == '__main__':
    unittest.main()
    



    
####### FIN ########
# %%
