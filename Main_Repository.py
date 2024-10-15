import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#####
import numpy as np
import numpy.fft as npf
import numpy.random as npr
import numpy.linalg as npl
import numba as nb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools 
import pywt
from datetime import datetime as dt
from astropy.stats import jackknife_stats, jackknife_resampling

####
from scipy.stats import sem, rankdata, pearsonr, spearmanr, chi2, mannwhitneyu, describe, ks_2samp, hmean, ttest_1samp, shapiro, normaltest 
from scipy.special import gamma, legendre, jv, softmax, expit, logit
from scipy.optimize import root, minimize, OptimizeResult
from scipy.integrate import trapezoid
from scipy.misc import derivative
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import butter, filtfilt, hilbert, tukey, argrelmax, hamming, cosine, savgol_filter, resample, welch, square, convolve2d, correlate2d
from scipy.linalg import orth

####sklearn
from sklearn.linear_model import LogisticRegression, Perceptron, LogisticRegressionCV
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, matthews_corrcoef, explained_variance_score, silhouette_samples, silhouette_score, brier_score_loss, log_loss
from sklearn.metrics import roc_curve, f1_score, precision_recall_curve, auc, average_precision_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, cosine_distances, pairwise_distances, rbf_kernel, chi2_kernel
from sklearn.model_selection import KFold, train_test_split, ShuffleSplit, LeaveOneOut
from sklearn.preprocessing import label_binarize, Normalizer, StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression

####
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN 

####
from tslearn.generators import random_walks, random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.shapelets import LearningShapelets
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.utils import to_sklearn_dataset

####
import tsfresh.feature_extraction.feature_calculators as tsfeat

###
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints

import sys

import warnings
warnings.filterwarnings('ignore')


class IsotonicCalibration:
    
    def __init__(self, Xtrain, Ytrain):
        
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        
        return
    
    def calibrate(self, score):

        ###define the Isotonic Regression
        IR = IsotonicRegression(y_min=0, y_max=1, out_of_bounds= 'clip')
        
        ###the fit the model
        IR.fit(self.Xtrain, self.Ytrain)
        
        return IR.predict(score)
    
class PlattCalibration:
    
    def __init__(self):
        
        return 
    
    def scale(self, coeffs, X):
        
        """Make Platt scaling given coefficients"""
        
        X0 = np.polyval(coeffs, X)
        return expit(X0)
    
    def get_coeffs(self, score):
            
        def _internal_(coeffs, score):
            
            ####make linear scaling
            score_scaled = np.polyval(coeffs, score)
            ####
            loss_ = np.mean(np.log(1+np.exp(score_scaled)))
            return loss_

        optimal_coeffs = minimize(_internal_, x0= [1, 1], args=(score))
        coeffs_ = optimal_coeffs.x
        return coeffs_
    
    def rescale(self, score):
        
        """Fit coeffs from scores and make Platt scaling"""
        
        COEFFS = self.get_coeffs(score)
        PROBS = self.scaling(COEFFS, score)
        return PROBS
    
    
class BetaCalibration:
    
    """ Use a Beta distribution to calibrate the ANN score with respect to the optimal AUROC threshold"""
    
    def __init__(self, scores, labels):
        
        self.X = scores
        self.Y = labels
    
        return
    
    ####
    def get_best_treshold(self):
        
        TP, FP, theta = roc_curve(self.Y, self.X, drop_intermediate=False)
        
        D = np.sqrt((1-TP)**2 + FP**2)
        D_theta = interp1d(theta, D, kind= 'quadratic', bounds_error=False, fill_value=(1, 1))
        
        theta_imputed = np.linspace(0, 1, 1000000)
        D_imputed = D_theta(theta_imputed)
        return theta_imputed[np.argmin(D_imputed)]
    
    def calibrate(self, scores):
        
        theta_imputed = self.get_best_treshold()
        theta = np.log(theta_imputed)
        alpha = np.log(.5)/theta
        prob = np.power(scores, alpha)
        return prob
    
    def predict_class(self, scores):
        
        PROBS = self.calibrate(scores)
        return (PROBS >= 5e-1).astype(int)
    
    
class ThresholdMovingDecision:
    
    """ Calibrate the ANN score by finding the optimal AUROC threshold"""
    
    def __init__(self, scores, labels):
        
        self.X = scores
        self.Y = labels
    
        return
    
    ####
    def get_best_treshold(self):
        
        TP, FP, theta = roc_curve(self.Y, self.X, drop_intermediate=False)
        D = np.sqrt((1-TP)**2 + FP**2)
        #D_theta = interp1d(theta, D, kind= 'linear', bounds_error=False, fill_value=(0, 1))
        #theta_imputed = np.linspace(0, 1, 1000000)
        #D_imputed = D_theta(theta_imputed)
        return theta[np.argmin(D)]
    
    def predict_class(self, scores):
        
        theta_best = self.get_best_treshold()
        return (scores >= theta_best).astype(int)
    
    
#######################################################################################
#######################################################################################
#######################################################################################
class Stochastic_Gradient_Descendent():
    
    """
    scipy.optimize-like implemendted SGD methods.
    
    This code is taken from "https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab"
    
    """
    
    def __init__(self):
        
        self.moo = 'MOO!'
        return 

    def sgd(
        self,
        fun,
        x0,
        jac,
        args=(),
        learning_rate=0.001,
        mass=0.9,
        startiter=0,
        maxiter=1000,
        callback=None,
        **kwargs):
        
        x = x0
        velocity = np.zeros_like(x)

        for i in range(startiter, startiter + maxiter):
            g = jac(x)

            if callback and callback(x):
                break

            velocity = mass * velocity - (1.0 - mass) * g
            x = x + learning_rate * velocity

        i += 1
        return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


    def rmsprop(
        fun,
        x0,
        jac,
        args=(),
        learning_rate=0.1,
        gamma=0.9,
        eps=1e-8,
        startiter=0,
        maxiter=1000,
        callback=None,
        **kwargs):
        """``scipy.optimize.minimize`` compatible implementation of root mean
        squared prop: See Adagrad paper for details.
        Adapted from ``autograd/misc/optimizers.py``.
        """
        x = x0
        avg_sq_grad = np.ones_like(x)

        for i in range(startiter, startiter + maxiter):
            g = jac(x)

            if callback and callback(x):
                break

            avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
            x = x - learning_rate * g / (np.sqrt(avg_sq_grad) + eps)

        i += 1
        return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


    def adam(
        fun,
        x0,
        jac,
        args=(),
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        startiter=0,
        maxiter=1000,
        callback=None,
        **kwargs):

        """``scipy.optimize.minimize`` compatible implementation of ADAM -
        [http://arxiv.org/pdf/1412.6980.pdf].
        Adapted from ``autograd/misc/optimizers.py``.
        """
        x = x0
        m = np.zeros_like(x)
        v = np.zeros_like(x)

        for i in range(startiter, startiter + maxiter):
            g = jac(x)

            if callback and callback(x):
                break

            m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1**(i + 1))  # bias correction.
            vhat = v / (1 - beta2**(i + 1))
            x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

        i += 1
        return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)



class DenseOrthogonal(tf.keras.constraints.Constraint):
    
    """Constraint to make Dense Layer weights Orthogonal (in the sense that W W.T = 1)
    
        NOTE: the total number of filters must always be larger than the time-lenght of filters!
    """
    
    def __init__(self):
        """No initial value..."""
        self.moo = "Hello!"

    def __call__(self, w):
        
        """... ... ..."""
        
        ##use numpy objects
        #numpW = w.numpy()
        numpW = K.eval(w)
        
        ####small trickle of noise
        delta= 1e-2
        noise_level = delta/(1-delta)
        NOISE = npr.wald(1, 1/noise_level, numpW.shape)
        
        ###diagonalization, use the orthonormal matrix of eigenvecotrs
        #eig_vals, eig_mtx = np.linalg.eig(numpW)
        
        
        ###convert into a tensor
        tensorW = tf.convert_to_tensor(orth(numpW*NOISE), dtype='float32')
        return tensorW

class SemiOrthogonal(tf.keras.constraints.Constraint):
    
    """Constrains weight tensors to be SemiOrthogonal (in the sense that W W.T = 1)
    
        NOTE: the total number of filters must always be larger than the time-lenght of filters!
    """
    
    def __init__(self):
        """No initial value..."""
        self.moo = "Hello!"
        
    def filter_as_mtx(self, FILTER):
    
        """Numpy Function: rearrange a filter in a 2-D array"""
    
        ###positions for arranging the filters
        list_points = list(itertools.product(range(FILTER.shape[0]), range(FILTER.shape[2])))

        ###display the filters a matrix
        ### NUMBER OF FILTERS X FILTER LENGHT
        weights_as_mtx = []
        for item in list_points:
            ii, jj = item
            weights_as_mtx.append(FILTER[ii, :, jj])
        return np.vstack(weights_as_mtx)

    def invert_filter_as_mtx(self, mtx, number_filters, number_features):
        
        """Numpy Function: Inverse of filter_as_mtx"""
        
        ###positions for arranging the filters
        list_points = list(itertools.product(range(number_filters), range(number_features)))

        ###display the filters a matrix
        ### NUMBER OF FILTERS X FILTER LENGHT
        mario = mtx.copy()
        filter_ = mario.reshape(number_filters, int(mtx.size/(number_filters*number_features)), number_features)
        for item in range(len(list_points)):
            ###
            ii, jj= list_points[item]
            filter_[ii, :, jj] = mtx[item]

        return filter_
    
    def Make_SemiOrthogonal(self, M, noise_level= 5e-2):
    
        """This function find a linear trasformation to make a generic matrix M semi-orthogonal.

        A--> generic linear trasformation (initial condition)
        Msquared --> MM^{T}.
        ***** ***** ***** ***** ***** 

        """

        #consider the shape of M
        Mshape = M.shape

        #noise_level
        delta = np.sqrt(noise_level/(1-noise_level))

        ####The internal LOSS function
        def __internal__(M, Mshape= Mshape):

            """Evaluate the loss function (tr(QQ^{T}))"""

            ###my generic linear transformation
            m = np.reshape(M, Mshape)
            ###
            Q= np.matmul(m, m.T)-np.identity(n= m.shape[0])
            ###
            return np.trace(np.matmul(Q.T, Q))
        ####

        ###initialize the Jacobian
        def __jac__(M, Mshape= Mshape):
            ###my generic linear transformation
            m = np.reshape(M, Mshape)
            ###
            Q= np.matmul(m, m.T)-np.identity(n= m.shape[0])
            
            return np.matmul(Q, m).ravel()
        
        solution = Stochastic_Gradient_Descendent().sgd(__internal__, 
                            x0=(M*npr.normal(1, delta, Mshape)).ravel(), 
                            jac = __jac__,
                            learning_rate=.5)

        return solution.x.reshape(Mshape)

    def __call__(self, w):
        
        """... ... ..."""
        
        ##use numpy objects
        #numpW = w.numpy()
        numpW = K.eval(w)
        
        
        ###find the semi-orthogonal
        numpF= self.filter_as_mtx(numpW) 
        #self.orthoF = numpF.copy()
        ###impose semi-orthogonality for each couple of "depating-arrival" node
        #for chunk in range(numpW.shape[0]):
        #    self.orthoF[chunk*numpW.shape[2]:(chunk+1)*numpW.shape[2]]= self.Make_SemiOrthogonal(self.orthoF[chunk*numpW.shape[2]:(chunk+1)*numpW.shape[2]])
        self.orthoF = self.Make_SemiOrthogonal(numpF)
        numpW_ = self.invert_filter_as_mtx(self.orthoF, 
                                           number_filters=numpW.shape[0], 
                                           number_features=numpW.shape[2])
        
        ###
        tensorW = tf.convert_to_tensor(numpW_, dtype='float32')
        return tensorW

    #def get_config(self):
    #    return {'ref_value': self.orthoF}
    
    
    
class Orthogonal(tf.keras.constraints.Constraint):
    
    """Constrains weight tensors to be Orthogonal (in the sense that W W.T = 1 as well as W W.T= 1)
    
        NOTE: the total number of filters must always be larger than the time-lenght of filters!
    """
    
    def __init__(self):
        """No initial value..."""
        self.moo = "Hello!"
        
    def filter_as_mtx(self, FILTER):
    
        """Numpy Function: rearrange a filter in a 2-D array"""
    
        ###positions for arranging the filters
        list_points = list(itertools.product(range(FILTER.shape[0]), range(FILTER.shape[2])))

        ###display the filters a matrix
        ### NUMBER OF FILTERS X FILTER LENGHT
        weights_as_mtx = []
        for item in list_points:
            ii, jj = item
            weights_as_mtx.append(FILTER[ii, :, jj])
        return np.vstack(weights_as_mtx)

    def invert_filter_as_mtx(self, mtx, number_filters, number_features):
        
        """Numpy Function: Inverse of filter_as_mtx"""
        
        ###positions for arranging the filters
        list_points = list(itertools.product(range(number_filters), range(number_features)))

        ###display the filters a matrix
        ### NUMBER OF FILTERS X FILTER LENGHT
        mario = mtx.copy()
        filter_ = mario.reshape(number_filters, int(mtx.size/(number_filters*number_features)), number_features)
        for item in range(len(list_points)):
            ###
            ii, jj= list_points[item]
            filter_[ii, :, jj] = mtx[item]

        return filter_
    
    def Make_Orthogonal(self, M, noise_level= 5e-2):
    
        """This function find a linear trasformation to make a generic matrix M orthogonal.

        Make Orthogonal via QR decomposition
        ***** ***** ***** ***** ***** 

        """

        delta = np.sqrt(1/(1-noise_level))
        noise_ = npr.normal(1, delta, M.shape)
        Mortho, __ = npl.qr(M*noise_)
        
        return Mortho

    def __call__(self, w):
        
        """... ... ..."""
        
        ##use numpy objects
        #numpW = w.numpy()
        numpW = K.eval(w)
        
        
        ###find the orthogonal set of weigths
        numpF= self.filter_as_mtx(numpW) 
        self.orthoF = self.Make_Orthogonal(numpF)
        numpW_ = self.invert_filter_as_mtx(self.orthoF, 
                                           number_filters=numpW.shape[0], 
                                           number_features=numpW.shape[2])
        
        ###
        tensorW = tf.convert_to_tensor(numpW_, dtype='float32')
        return tensorW

    #def get_config(self):
    #    return {'ref_value': self.orthoF}


class signals_from_FS():
    
    def __init__(self):
        self.hello = 'HELLO!'
        return
    
    def gamma_FS(self, Nfreqs= 100, period=1 , Time_interval= [0, 1], 
                 Time_points= 128, a0= 0, scale= 8, shape= 2, phase= 0):

        ### ### ###
        ### define the cosine coeffs    
        ###amplitude
        ns = 1+np.arange(Nfreqs)
        coeffs = ns**(shape-1)*np.exp(-ns/scale)/(gamma(shape)*scale**(shape))
        #coeffs = coeffs/(coeffs**2).sum()

        T0, T1 = Time_interval
        Time = np.linspace(T0, T1, Time_points)
        Phase= phase*period
        Z = np.array([coeffs[kk-1]*np.sin(2*np.pi*(Time+Phase)*kk/period) for kk in ns])
        Signal = np.array(Z).sum(axis= 0)+.5*a0
        erg = np.mean(Signal**2)
        return Signal/Signal.std()


    def powerlaw_FS(self, Nfreqs= 100, period=1 , Time_interval= [0, 1], Time_points= 128, 
                    a0= 0, shape= -2., loc= 4, phase= 0):

        ### ### ###
        ### define the cosine coeffs    
        ###amplitude
        ns = 1+np.arange(Nfreqs)
        coeffs = (ns+loc)**(shape)
        #coeffs = coeffs/(coeffs**2).sum()

        T0, T1 = Time_interval
        Time = np.linspace(T0, T1, Time_points)
        Phase= phase*period
        Z = np.array([coeffs[kk-1]*np.sin(2*np.pi*(Time+Phase)*kk/period) for kk in ns])
        Signal = np.array(Z).sum(axis= 0)+.5*a0
        erg = np.mean(Signal**2)
        return Signal/Signal.std()


    def gaussian_FS(self, Nfreqs= 100, period=1 , Time_interval= [0, 1], Time_points= 128, 
                    a0= 0, scale= 4, loc= 16, phase= 0):

        ### ### ###
        ### define the cosine coeffs    
        ###amplitude
        ns = 1+np.arange(Nfreqs)
        coeffs = np.exp(-((loc-ns)/scale)**2)
        #coeffs = coeffs/(coeffs**2).sum()

        T0, T1 = Time_interval
        Time = np.linspace(T0, T1, Time_points)
        Phase= phase*period
        Z = np.array([coeffs[kk-1]*np.sin(2*np.pi*(Time+Phase)*kk/period) for kk in ns])
        Signal = np.array(Z).sum(axis= 0)+.5*a0
        erg = np.mean(Signal**2)
        return Signal/Signal.std()

    def beta_FS(self, Nfreqs= 100, period=1 , Time_interval= [0, 1], Time_points= 128, 
                a0= 0, shape1= 3., shape2= 1., phase= 0):

        ### ### ###
        ### define the cosine coeffs    
        ###amplitude
        ns = 1+np.arange(Nfreqs)
        ns_max = ns.max()
        coeffs = ((ns/ns_max)**(shape1-1))*((1-ns/ns_max)**(shape2-1))
        #coeffs = coeffs/(coeffs**2).sum()
        
        T0, T1 = Time_interval
        Time = np.linspace(T0, T1, Time_points)
        Phase= phase*period
        Z = np.array([coeffs[kk-1]*np.sin(2*np.pi*(Time+Phase)*kk/period) for kk in ns])
        Signal = np.array(Z).sum(axis= 0)+.5*a0
        erg = np.mean(Signal**2)
        return Signal/Signal.std()

    def bessel_FS(self, Nfreqs= 100, period=1 , Time_interval= [0, 1], Time_points= 128, 
                a0= 0, order= 16, phase= 0):

        ### ### ###
        ### define the cosine coeffs    
        ###amplitude
        ns = 1+np.arange(Nfreqs)
        coeffs = jv(order, ns)
        #coeffs = coeffs/(coeffs**2).sum()
        
        T0, T1 = Time_interval
        Time = np.linspace(T0, T1, Time_points)
        Phase= phase*period
        Z = np.array([coeffs[kk-1]*np.sin(2*np.pi*(Time+Phase)*kk/period) for kk in ns])
        Signal = np.array(Z).sum(axis= 0)+.5*a0
        erg = np.mean(Signal**2)
        return Signal/Signal.std()
    
    def gen_instances(self, population_size= 1000, Time_points = 128, separation= 1, minority_ratio= .5):
        
        ####generate classes
        classes = npr.choice([0, 1], size= population_size, p= [minority_ratio, 1-minority_ratio])
        
        ###define phases
        phases= np.array([(.5-item)*npr.random() for item in classes])
        
        ###
        X = []
        for kk in range(population_size):
            ph = np.clip(npr.random(5), a_min=1e-8, a_max=1-1e-8)
            if classes[kk] == 0:
                phases = ph*.5
            elif classes[kk]==1:
                phases = -ph*.5
                
            ####
            separation_coin = npr.random()
            if separation_coin<= np.clip(1-separation, a_min=0, a_max=1):
                phases *= -phases
                
            X.append(np.array([self.gamma_FS(phase=phases[0], Time_points=Time_points), 
                     self.powerlaw_FS(phase=phases[1], Time_points=Time_points), 
                    self.gaussian_FS(phase=phases[2], Time_points=Time_points), 
                    self.beta_FS(phase=phases[3], Time_points=Time_points), 
                    self.bessel_FS(phase=phases[4], Time_points=Time_points)]).T)
            
        ### ### ### ### ###
        return np.array(X), classes
    
    def gen_instance_NoPhase(self):
        
        ####generate classes
        
                
        X = np.vstack(np.array([self.gamma_FS(), 
                     self.powerlaw_FS(), 
                    self.gaussian_FS(), 
                    self.beta_FS(), 
                    self.bessel_FS()]).T)
            
        ### ### ### ### ###
        return X

class RAF(tf.keras.layers.Layer):
    
    def __init__(self, degree= 3, kind= 'sigmoidal'):
        super(RAF, self).__init__()
        
        Init_ = tf.random_normal_initializer(mean= 0, stddev= 1)
        
        if kind == 'sigmoidal':
            self.alpha = [tf.Variable(initial_value= Init_(shape=(1,), dtype="float32"), trainable=True) for kk in range(degree+1)]
            self.beta = [tf.Variable(initial_value= Init_(shape=(1,), dtype="float32"), trainable=True) for kk in range(degree+1)]
        elif kind == 'hockey-stick':
            self.alpha = [tf.Variable(initial_value= Init_(shape=(1,), dtype="float32"), trainable=True) for kk in range(degree+1)]
            self.beta = [tf.Variable(initial_value= Init_(shape=(1,), dtype="float32"), trainable=True) for kk in range(degree)]
        elif kind =='bumped':
            self.alpha = [tf.Variable(initial_value= Init_(shape=(1,), dtype="float32"), trainable=True) for kk in range(degree)]
            self.beta = [tf.Variable(initial_value= Init_(shape=(1,), dtype="float32"), trainable=True) for kk in range(degree+1)]
        
        
    def call(self, inputs):
        
        P = tf.math.polyval(self.alpha, inputs)
        Q = tf.math.polyval(self.beta, inputs)
        return P/Q
    
class RationalLayer(tf.keras.layers.Layer):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are learned array with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the weights of the numerator P.
        beta_initializer: initializer function for the weights of the denominator Q.
        alpha_regularizer: regularizer for the weights of the numerator P.
        beta_regularizer: regularizer for the weights of the denominator Q.
        alpha_constraint: constraint for the weights of the numerator P.
        beta_constraint: constraint for the weights of the denominator Q.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """

    def __init__(self, alpha_initializer=[1.1915, 1.5957, 0.5, 0.0218], beta_initializer=[2.383, 0.0, 1.0], 
                 alpha_regularizer=None, beta_regularizer=None, alpha_constraint=None, beta_constraint=None,
                 shared_axes=None, **kwargs):
        super(RationalLayer, self).__init__(**kwargs)
        self.supports_masking = True

        # Degree of rationals
        self.degreeP = len(alpha_initializer) - 1
        self.degreeQ = len(beta_initializer) - 1
        
        # Initializers for P
        self.alpha_initializer = [initializers.Constant(value=alpha_initializer[i]) for i in range(len(alpha_initializer))]
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        
        # Initializers for Q
        self.beta_initializer = [initializers.Constant(value=beta_initializer[i]) for i in range(len(beta_initializer))]
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        
        self.coeffsP = []
        for i in range(self.degreeP+1):
            # Add weight
            alpha_i = self.add_weight(shape=param_shape,
                                     name='alpha_%s'%i,
                                     initializer=self.alpha_initializer[i],
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
            self.coeffsP.append(alpha_i)
            
        # Create coefficients of Q
        self.coeffsQ = []
        for i in range(self.degreeQ+1):
            # Add weight
            beta_i = self.add_weight(shape=param_shape,
                                     name='beta_%s'%i,
                                     initializer=self.beta_initializer[i],
                                     regularizer=self.beta_regularizer,
                                     constraint=self.beta_constraint)
            self.coeffsQ.append(beta_i)
        
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
                    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
                    self.built = True

    def call(self, inputs, mask=None):
        # Evaluation of P
        outP = tf.math.polyval(self.coeffsP, inputs)
        # Evaluation of Q
        outQ = tf.math.polyval(self.coeffsQ, inputs)
        # Compute P/Q
        out = tf.math.divide(outP, outQ)
        return out

    def get_config(self):
        config = {
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(RationalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
class DataTraining:
    
    """"
    Select the training dataset
    
    1. Random Under Sampling -- select a portion of data
    2. use SMOTE to augment the dataset
    """
    
    def __init__(self, X, Y, ihs_ratio= 25e-2, smote_ratio= 1):
        
        ## X --> the data to utilize
        
        self.X = X
        self.Y = Y
        
    def make_datatraining(self):
        
        #
        nrus0 = np.minimum(self.X[self.Y==0].shape[0], self.max_size-int(self.Y.sum()))
        RUS= InstanceHardnessThreshold(estimator=LogisticRegression, 
                                       sampling_strategy=ihs_ratio, 
                                random_state=58)
        #
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=58)
        
        ###
        xx, yy = RUS.fit_resample(self.X, self.Y)
        ###
        xx, yy = smote.fit_resample(xx, yy)
        
        return xx, yy
    
    
class TimeSeriesPCA:
    
    def __init__(self, X, ncomponents= 10):
        
        ##the data
        self.shape_tensor = X.shape
        self.X = X.reshape(X.shape[0], -1)
        
        ###fit PCA
        self.pca = PCA(n_components= ncomponents)
        self.pca.fit(self.X)
        
        self.explain_variance_ratio = self.pca.explained_variance_ratio_.sum()
    
    def restore_shape(self, X):
        return X.reshape(-1, self.shape_tensor[1], self.shape_tensor[2])    
    
    def flat_shape(self, X):
        return X.reshape(self.shape_tensor[0], -1)
    
    def transform(self, X):
        return self.pca.transform(X)
    
    def anti_transform(self, X):
        return self.pca.inverse_transform(X)
    
    
    
class CNN_1D():
    
    def __init__(self):
        self.hello = 'HELLO!'
        return
    
    def Rational_fixed(self, x):
        
        """Rational Fixed Activation Function"""
        
        
        return (K.pow(x, 3)-K.pow(x, 1))/(K.pow(x, 2)-K.pow(x, 1)+1)
    
    def CNN_1D(self,
                X,
                filters= 8, 
                 kernel_size= 5, 
                 poolsize= 2,
                 DropOut= 50e-2,
                 strides= 1, 
                 activation= 'relu', 
                 padding = 'valid', 
                 bias = False, 
                 activation_pred= 'sigmoid', 
                 deepness= 3, 
                 dense_units= 1,      
                 lr = 1e-3, 
                 semi_ortho_constraint = True, 
                 print_summary= False):
        
        #######################################
        ###Define the CNN model
        ######################################
        
        ### Input
        Inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2]))

        ### ### ### ### ###
        ### 1st_layer ## ##
        if semi_ortho_constraint:
            X = tf.keras.layers.Conv1D(filters= filters, 
                            kernel_size= kernel_size,
                            activation= None, 
                            strides = strides, 
                            padding = padding, 
                            use_bias = False,
                            kernel_constraint= SemiOrthogonal())(Inputs)
        else:
            X = tf.keras.layers.Conv1D(filters= filters, 
                            kernel_size= kernel_size,
                            activation= None, 
                            strides = strides, 
                            padding = padding, 
                            use_bias = False)(Inputs)
        
        if activation =='Rational':
            X = RationalLayer()(X)
        elif activation == 'rational_sigmoid':
            X = RAF(kind='sigmoid')(X)
        elif activation == 'rational_hockey':
            X = RAF(kind='hockey-stick')(X)
        elif activation == 'rational_bumped':
            X = RAF(kind='bumped')(X)
        elif activation == 'FFNN':
            # FFNN --> FeedForwardNeuralNetwork
            X = tf.keras.layers.Dense(units= filters, use_bias= False, activation= None)(X)
            X = tf.keras.layers.Activation('relu')(X)
            X = tf.keras.layers.Dense(units= filters, use_bias= False, activation= None)(X)
            X = tf.keras.layers.Activation('relu')(X)
        else:
            X = tf.keras.layers.Activation(activation)(X)
        
        X = tf.keras.layers.MaxPool1D(pool_size = poolsize, padding= 'valid')(X)
        X = tf.keras.layers.GaussianDropout(rate = DropOut)(X)

        ###other layers
        for jj in range(1, deepness):
            if semi_ortho_constraint:
                X = tf.keras.layers.Conv1D(filters= filters, 
                            kernel_size= kernel_size,
                            activation= None, 
                            strides = strides, 
                            padding = padding, 
                            use_bias = False,
                            kernel_constraint= SemiOrthogonal())(X)
            else:
                X = tf.keras.layers.Conv1D(filters= filters, 
                            kernel_size= kernel_size,
                            activation= None, 
                            strides = strides, 
                            padding = padding, 
                            use_bias = False)(X)
            
            
            
            if activation =='Rational':
                X = RationalLayer()(X)
            elif activation == 'rational_sigmoid':
                X = RAF(kind='sigmoid')(X)
            elif activation == 'rational_hockey':
                X = RAF(kind='hockey-stick')(X)
            elif activation == 'rational_bumped':
                X = RAF(kind='bumped')(X)
            elif activation == 'FFNN':
                # FFNN --> FeedForwardNeuralNetwork
                X = tf.keras.layers.Dense(units= filters, use_bias= False, activation= None)(X)
                X = tf.keras.layers.Activation('relu')(X)
                X = tf.keras.layers.Dense(units= filters, use_bias= False, activation= None)(X)
                X = tf.keras.layers.Activation('relu')(X)
            else:
                X = tf.keras.layers.Activation(activation)(X)
            #X = MUAF()(X)
            #X = RationalLayer()(X)
            X = tf.keras.layers.MaxPool1D(pool_size = poolsize, padding= 'valid')(X)
            X = tf.keras.layers.GaussianDropout(rate = DropOut)(X)

        ### flattern   
        X = tf.keras.layers.Flatten()(X)

        ###make final prediction (with Platt's calibration)
        #Xfinal = tf.keras.layers.Dense(units = dense_units, activation= 'sigmoid', use_bias= True)(X) 
        Xfinal = tf.keras.layers.Dense(units = dense_units, activation= 'linear', use_bias= True)(X) 
        if dense_units >= 2:
            Xfinal = tf.keras.layers.Activation('softmax')(Xfinal)
        else:
            Xfinal = tf.keras.layers.Activation('sigmoid')(Xfinal)

        ##define model
        mymodel = tf.keras.models.Model(Inputs, Xfinal)
        #print(mymodel.summary())

        ###PRINT MODEL's SUMMARY
        if print_summary:
            print(mymodel.summary())

        ### Optimizer ADAM+SCE
        adam = tf.keras.optimizers.Adam(learning_rate= lr)
        bce = tf.keras.losses.BinaryCrossentropy()
        sce = tf.keras.losses.SparseCategoricalCrossentropy()
        cce = tf.keras.losses.CategoricalCrossentropy()
        
        if dense_units == 1:
            mymodel.compile(optimizer= adam,
                        loss = bce)
        else:
            mymodel.compile(optimizer= adam,
                        loss = sce)
        
        return mymodel
    
    def use_SMOTEENN(self, X, Y):
            
        """ Use SMOTE to augment training and validation data"""
           
        ###AUGMENT VALIDATION DATA
        smoteenn = SMOTEENN(sampling_strategy= 'auto')

        ####
        Xval_pca = to_sklearn_dataset(X)
        #
        pca= PCA(n_components= int(.8*min(Xval_pca.shape[0], Xval_pca.shape[1])))
        Xval_pca = pca.fit_transform(Xval_pca)
        Xval_pca, Y = smoteenn.fit_resample(Xval_pca, Y)
        Xval_pca = pca.inverse_transform(Xval_pca)
        X = Xval_pca.reshape(Xval_pca.shape[0], X.shape[1], X.shape[2]) 
        
        return X, Y


    def validate_model(self, X, Y, model, 
                           cv= 5,
                           use_SMOTEENN = False,
                           risk_score = False,
                           intermediate_results = True,
                           rate_decay = 1e-7, 
                           batch_size= 32, 
                           epochs= 1000, 
                           verbose= 1, 
                           patience= 5):
        
        ####model to fit
        #model_fit = tf.keras.models.clone_model(model)
        
        ###define Kfold-Cross-Validation
        kfold = KFold(cv, shuffle=True, random_state= 58) ###58 --> 'O paccotto
        
        ###early_stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0, 
                                        patience= patience)
        
        ##lr_scheduler
        def LRScheduler(epoch, lr= model.optimizer.lr.numpy(), rate_decay= rate_decay):
            ### linear decreasing
            lrate = lr -rate_decay 
            return lrate
        lrate = tf.keras.callbacks.LearningRateScheduler(LRScheduler)
        
        if use_SMOTEENN:
            
            """ Use SMOTE to augment training and validation data"""
            
            ### 85 means "O' contropaccotto 
            #Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, shuffle= True, train_size=.5, random_state=85) 
        
            ###AUGMENT VALIDATION DATA
            smoteenn = SMOTEENN(sampling_strategy= 'auto')

            ####
            Xval_pca = to_sklearn_dataset(X)
            #
            pca= PCA(n_components= int(.8*min(Xval_pca.shape[0], Xval_pca.shape[1])))
            Xval_pca = pca.fit_transform(Xval_pca)
            Xval_pca, Y = smoteenn.fit_resample(Xval_pca, Y)
            Xval_pca = pca.inverse_transform(Xval_pca)
            X = Xval_pca.reshape(Xval_pca.shape[0], X.shape[1], X.shape[2]) 
            
        ### 85 means "O' contropaccotto 
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, shuffle= True, train_size=.8, random_state=85) 
        
        ###metrics
        auc = np.zeros(cv)
        auc_test = np.zeros(cv)
        auprc = np.zeros(cv)
        auprc_test = np.zeros(cv)
        mcc = np.zeros(cv)
        mcc_test = np.zeros(cv)
        phi = []
        
        Xuse, Xtest, Yuse, Ytest = train_test_split(X, Y, shuffle= True, train_size=.8, random_state=85) ###O' contropaccotto is 85 
       
    
        ####weights of the model (before .fit)
        weights = model.get_weights()
        
        fold = 0
        for index_train, index_test in kfold.split(Xuse):
                        
            ###train-test split
            Xtrain, Xval = Xuse[index_train], Xuse[index_test]
            Ytrain, Yval = Yuse[index_train], Yuse[index_test]
            
            #Utest = np.hstack([Ytest.reshape(-1, 1), 1-Ytest.reshape(-1, 1)])
            
            ### call fit
            model.fit(Xtrain, Ytrain, 
                      validation_data = (Xval, Yval),
                      epochs= epochs, 
                      batch_size= batch_size, 
                      verbose= verbose,
                      callbacks=[lrate, early_stopping])
            
            
            ###whether the output is non-probabilistic "risk score" -- VALIDATION
            if risk_score:
                
                ###evaluate the risk score
                score_train = model.predict(Xtrain, verbose= None).ravel()
                score_val = model.predict(Xval, verbose= None).ravel()
                
                ### Fit the ISOTONIC CALIBRATION
                IsoCal = IsotonicCalibration(score_train, Ytrain)
                ##estimate the calibrated scores (i.e., what we mean for probability)
                prob_val = IsoCal.calibrate(score_val)
                ###
                auc_ = roc_auc_score(y_score= prob_val, y_true= Yval)
                mcc_ = matthews_corrcoef(y_pred=prob_val>.5, y_true= Yval)
                auprc_ = average_precision_score(y_score= prob_val, y_true= Yval)
            else:
                score_train = model.predict(Xtrain, verbose= None)[:, -1]
                score_val = model.predict(Xval, verbose= None)[:, -1]
                ###
                auc_ = roc_auc_score(y_score= score_val, y_true= Yval)
                mcc_ = matthews_corrcoe(y_pred=score_val>.5, y_true= Yval)
                auprc_ = average_precision_score(y_score= score_val, y_true= Yval)
            
            ###whether the output is non-probabilistic "risk score" -- TEST
            if risk_score:
                
                ###evaluate the risk score
                score_train = model.predict(Xtrain, verbose= None).ravel()
                score_test = model.predict(Xtest, verbose= None).ravel()
                
                ### Fit the ISOTONIC CALIBRATION
                IsoCal = IsotonicCalibration(score_train, Ytrain)
                ##estimate the calibrated scores (i.e., what we mean for probability)
                prob_test = IsoCal.calibrate(score_test)
                ###
                auc_test_= roc_auc_score(y_score= prob_test, y_true= Ytest)
                mcc_test_ = matthews_corrcoef(y_pred=prob_test>.5, y_true= Ytest)
                auprc_test_ = average_precision_score(y_score= prob_test, y_true= Ytest)
            else:
                score_test = model.predict(Xtest, verbose= None)[:, -1]
                ###
                auc_test_ = roc_auc_score(y_score= score_test, y_true= Ytest)
                mcc_test_ = matthews_corrcoe(y_pred=score_test>.5, y_true= Ytest)
                auprc_test_ = average_precision_score(y_score= score_test, y_true= Ytest)
            
                
            ####print all results
            if intermediate_results:
                print("**** **** **** **** **** ****")
                print("***** FOLD: ", fold , "*********")
                print('Partial AUC (validation):', np.round(auc_, 3))
                print('Partial AUC (test):', np.round(auc_test_, 3))
                print('Partial MCC (validation):', np.round(mcc_, 3))
                print('Partial MCC (test):', np.round(mcc_test_, 3))
                print('Partial AUPRC (test):', np.round(mcc_test_, 3))
                print('Partial AUPRC (validation):', np.round(mcc_test_, 3))
                print('***** ***** ***** ***** *****')
            
            ######
            ###after fitting delete the training and validation variables
            del Xtrain
            del Xval
            del Ytrain
            del Yval
            
            
            ####save the results
            auc[fold] = auc_.round(3)
            mcc[fold] = np.round(mcc_, 3)
            auprc[fold] = auprc_.round(3)
            #
            auc_test[fold] = auc_test_.round(3)
            mcc_test[fold] = np.round(mcc_test_, 3)
            auprc_test[fold] = np.round(auprc_test_, 3)
            
            fold += 1
            #phi.append(phi_)
            
            #### RESET WEIGHTS
            model.set_weights(weights)
            
            ###CLEAR MODEL
            #K.clear_session()
            
            ###delete varibale model...
            #del model_fit
                        
        ######
        #print('**** **** **** **** ****')
        #print('AUC (Validation):', np.mean(auc).round(2), np.round(sem(auc), 2))
        #print('MCC (Validation):', np.mean(mcc).round(2), np.round(sem(mcc), 2))
        #print('AUPRC (Validation):', np.mean(auprc).round(2), np.round(sem(auprc), 2)) 
        #print('AUC (Test):', np.mean(auc_test).round(2), np.round(sem(auc_test), 2))
        #print('MCC (Test):', np.mean(mcc_test).round(2), np.round(sem(mcc_test), 2))
        #print('AUPRC (test):', np.mean(auprc_test).round(2), np.round(sem(auprc_test), 2)) 
        
        #print('Phi:', np.mean(phi).round(2), sem(phi).round(2))
        
        ### ### ###
        jack_result_auc = jackknife_stats(np.array(auc), np.mean)
        jack_result_mcc = jackknife_stats(np.array(mcc), np.mean)
        jack_result_auprc = jackknife_stats(np.array(auprc), np.mean)
        jack_result_prodof3_raw = jackknife_stats(np.array(auc)*np.array(mcc)*np.array(auprc), np.mean)
        jack_result_prodof3 = jackknife_stats(np.array(auc)*.5*(1+np.array(mcc))*np.array(auprc), np.mean)
        
        
        result = {'AUC': jack_result_auc[0].round(2),
                  'AUC_err': np.maximum(0.01, np.round(jack_result_auc[2], 2)),
                  'MCC': jack_result_mcc[0].round(2),
                  'MCC_err': np.maximum(0.01, np.round(jack_result_mcc[2], 2)),
                  'AUPRC': jack_result_auprc[0].round(2),
                  'AUPRC_err': np.maximum(np.round(jack_result_auprc[2], 2), 0.01),
                  "3MCS_raw": jack_result_prodof3_raw[0].round(2),
                  "3MCS_raw_err":np.maximum(np.round(jack_result_prodof3_raw[2], 2), 0.01),
                  "3MCS": jack_result_prodof3[0].round(2),
                  "3MCS_err": np.maximum(np.round(jack_result_prodof3[2], 2), 0.01),
                  'Test_Data': (Xtest, Ytest)}
        
        return result
    
    
    def one_held_out_validation(self, 
                                X, 
                                Y, 
                                model, 
                                use_SMOTEENN = False,
                                risk_score = False, 
                                rate_decay = 1e-7, 
                                batch_size= 32, 
                                epochs= 1000, 
                                verbose= 1, 
                                patience= 5):
        
        ####model to fit
        #model_fit = tf.keras.models.clone_model(model)
        
        
        
        ###callbacks
        ###early_stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0, 
                                        patience= patience)
        
        ##lr_scheduler
        def LRScheduler(epoch, lr= model.optimizer.lr.numpy(), rate_decay= rate_decay):
            ### linear decreasing
            lrate = lr -rate_decay 
            return lrate
        lrate = tf.keras.callbacks.LearningRateScheduler(LRScheduler)
        
        if use_SMOTEENN:
            
            """ Use SMOTE to augment training and validation data"""
            
            ### 85 means "O' contropaccotto 
            #Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, shuffle= True, train_size=.5, random_state=85) 
        
            ###AUGMENT VALIDATION DATA
            smoteenn = SMOTEENN(sampling_strategy= 'auto')

            ####
            Xval_pca = to_sklearn_dataset(X)
            #
            pca= PCA(n_components= int(.8*min(Xval_pca.shape[0], Xval_pca.shape[1])))
            Xval_pca = pca.fit_transform(Xval_pca)
            Xval_pca, Y = smoteenn.fit_resample(Xval_pca, Y)
            Xval_pca = pca.inverse_transform(Xval_pca)
            X = Xval_pca.reshape(Xval_pca.shape[0], X.shape[1], X.shape[2]) 
            
        ### 85 means "O' contropaccotto 
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, shuffle= True, train_size=.8, random_state=85) 
        
        ### call fit
        model.fit(Xtrain, Ytrain, 
                      validation_data = (Xtest, Ytest),
                      epochs= epochs, 
                      batch_size= batch_size, 
                      verbose= verbose,
                      callbacks=[lrate, early_stopping])
            
            
        ###whether the output is non-probabilistic "risk score"
                
            
        ####
        if risk_score:
            ### use Isotonic Regression to make score more familiar with a probabilistic interpretation
            score_train = model.predict(Xtrain).ravel()
            IsoCal = IsotonicCalibration(score_train, Ytrain)
            ###
            score_test = model.predict(Xtest).ravel()
            prob_test = IsoCal.calibrate(score_test)
            ###
            auc_test_ = roc_auc_score(y_score= prob_test, y_true= Ytest)
            auprc_test_ = average_precision_score(y_score= prob_test, y_true= Ytest)
            ###
            y_pred_class = (prob_test>=.5).astype(int) 
            mcc_test_ = matthews_corrcoef(y_true = y_pred_class, y_pred = Ytest)
            
            print('Partial AUROC (test):', np.round(auc_test_, 3))
            print('Partial AUPRC (test):', np.round(auprc_test_, 3))
            print('Partial MCC (test):', np.round(mcc_test_, 3))
            print('***** ***** ***** ***** *****')
        
        else:
            score_test = model.predict(Xtest)[:, 1]
                
            #TMD_test = ThresholdMovingDecision(score_test, Ytest)
            auc_test_ = roc_auc_score(y_score= score_test, y_true= Ytest)
            auprc_test_ = average_precision_score(y_score= score_test, y_true= Ytest)
                
            print('Partial AUROC (test):', np.round(auc_test_, 3))
            print('Partial AUPRC (test):', np.round(auprc_test_, 3))
            print('***** ***** ***** ***** *****')
        
            
            
            
        result = {'AUROC': auc_test_,
                  'AUPRC': auprc_test_,
                  'Test_Data': (Xtest, Ytest)}
        
        return result
    
    def validate_model_RUSROS(self, X, Y, model, risk_score = False, use_extradata= False, xleft= None, yleft= None, cv= 5, 
                       rate_decay = 1e-7, 
                          batch_size= 32, epochs= 1000, verbose= 1, patience= 5, warm_start= False, time_series_data = False, 
                             iht_sampling_strategy = 1):
        
        
        """
        Validate the model by constrasting data imbalancement.
        Use Random Under Sampling and Random Over Sampling techniques together
        """
        
        ####model to fit
        #model_fit = tf.keras.models.clone_model(model)
        
        ###define Kfold-Cross-Validation
        kfold = KFold(cv, shuffle=True, random_state= 58) ###58 --> 'O paccotto
        
        ###callbacks
        ###early_stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0, 
                                        patience= patience)
        
        ##lr_scheduler
        def LRScheduler(epoch, lr= model.optimizer.lr.numpy(), rate_decay= rate_decay):
            ### linear decreasing
            lrate = lr -rate_decay 
            return lrate
        lrate = tf.keras.callbacks.LearningRateScheduler(LRScheduler)
        
        ###metrics
        auc = np.zeros(cv)
        auc_test = np.zeros(cv)
        auprc = np.zeros(cv)
        auprc_test = np.zeros(cv)
        phi = []
        
        #### #### #### ####
        ###GET Pre-Training Data
        ####
        #rus = RandomUnderSampler(sampling_strategy= 1/8, random_state= 58)

        ####
        #iht = InstanceHardnessThreshold(estimator= LogisticRegression(), 
        #                                    sampling_strategy= iht_sampling_strategy, 
        #                                    random_state= 58)

        ####
        #X_pca = to_sklearn_dataset(X)
        #
        #pca= PCA(n_components= int(.8*min(X_pca.shape[0], X_pca.shape[1])))
        #X_pca = pca.fit_transform(X_pca)
        #X_pca, Y = iht.fit_resample(X_pca, Y)
        #X_pca = pca.inverse_transform(X_pca)
        #X = X_pca.reshape(X_pca.shape[0], X.shape[1], X.shape[2]) 
            
        
        #### #### #### PRE-TRAIN #### #### #### ####
        #model.fit(Xpre, Ypre, 
        #          validation_data = (Xuse, Yuse),
        #          epochs= epochs, 
        #          batch_size= batch_size, 
        #          verbose= verbose,
        #          callbacks=[lrate, early_stopping])
        
        ####take the random 50% for test set
        Xuse, Xtest, Yuse, Ytest = train_test_split(X, Y, shuffle= True, test_size=.50, random_state=85)
        
        
        ##################################################################################################    
                
        ####weights of the model (before .fit)
        weights = model.get_weights()
        
            
        fold = 0
        for index_train, index_test in kfold.split(Xuse):
            
            ###train-test split
            Xtrain, Xval = Xuse[index_train], Xuse[index_test]
            Ytrain, Yval = Yuse[index_train], Yuse[index_test] 
            
            ###REDUCE TRAINING DATA
            ####
            #iht = InstanceHardnessThreshold(estimator= LogisticRegression(), 
            #                                sampling_strategy= 1, 
            #                                random_state= 58)

            ####
            #Xtrain_pca = to_sklearn_dataset(Xtrain)
            #
            #pca= PCA(n_components= int(.8*min(Xtrain_pca.shape[0], Xtrain_pca.shape[1])))
            #Xtrain_pca = pca.fit_transform(Xtrain_pca)
            #Xtrain_pca, Ytrain = iht.fit_resample(Xtrain_pca, Ytrain)
            #Xtrain_pca = pca.inverse_transform(Xtrain_pca)
            #Xtrain = Xtrain_pca.reshape(Xtrain_pca.shape[0], Xtrain.shape[1], Xtrain.shape[2]) 
            
            ###AUGMENT VALIDATION DATA
            #smote = SMOTE(sampling_strategy= 1, random_state= 58)

            ####
            #Xval_pca = to_sklearn_dataset(Xval)
            #
            #pca= PCA(n_components= int(.8*min(Xval_pca.shape[0], Xval_pca.shape[1])))
            #Xval_pca = pca.fit_transform(Xval_pca)
            #Xval_pca, Yval = smote.fit_resample(Xval_pca, Yval)
            #Xval_pca = pca.inverse_transform(Xval_pca)
            #Xval = Xval_pca.reshape(Xval_pca.shape[0], Xval.shape[1], Xval.shape[2]) 
            
           
            ######use SMOTE for FOR XTRAIN
            #smote = SMOTE(sampling_strategy= 1)
            #if time_series_data:
                
                ###get the sufficient number of components
            #    ncomponents= int(.15*Xtrain.shape[1]*Xtrain.shape[2])
            #    rate_components = int(.05*Xtrain.shape[1]*Xtrain.shape[2])

                ###
            #    if ncomponents>= Xtrain.shape[0]:
            #        TSpca = TimeSeriesPCA(Xtrain, Xtrain.shape[0])
            #    else: 
            #        TSpca = TimeSeriesPCA(Xtrain, ncomponents)
            #        exp_var = TSpca.explain_variance_ratio
            #        while exp_var < .95:
            #            ncomponents += rate_components 
            #            if ncomponents<= Xtrain.shape[0]:
            #                break
            #            TSpca = TimeSeriesPCA(Xtrain, ncomponents)
            #            exp_var = TSpca.explain_variance_ratio
            #            print(exp_var, ncomponents)

                ### Use the PCA latent data to geenrate the new trainig data space
            #    Xflat = TSpca.flat_shape(Xtrain)
            #    Xflat = TSpca.transform(Xflat)

                ###contrast data imbalancement (with Over-sampling)
            #    Xtrain, Ytrain = smote.fit_resample(Xflat, Ytrain) 

                ####
            #    Xtrain = TSpca.anti_transform(Xtrain)
            #    Xtrain = TSpca.restore_shape(Xtrain)

            #else:
            
                ###make resampling
            #    smote.fit(Xtrain, Ytrain)
            #    Xtrain, Ytrain = smote.fit_resample(Xtrain, Ytrain)

                     
            ### call fit
            model.fit(Xtrain, Ytrain, 
                      validation_data = (Xval, Yval),
                      epochs= epochs, 
                      batch_size= batch_size, 
                      verbose= verbose,
                      callbacks=[lrate, early_stopping])
            
            
            ###whether the output is non-probabilistic "risk score"
            if risk_score:
                
                ###evaluate the risk score
                score_train = model.predict(Xtrain).ravel()
                score_val = model.predict(Xval).ravel()
                
                ### Fit the ISOTONIC CALIBRATION
                IsoCal = IsotonicCalibration(score_train, Ytrain)
                ##estimate the calibrated scores (i.e., what we mean for probability)
                prob_val = IsoCal.calibrate(score_val)
                ###
                auc_ = roc_auc_score(y_score= prob_val, y_true= Yval)
                ###precision_recall_area_under_the _curve
                #precision_, recall_, __ = precision_recall_curve(y_true= Yval, probas_pred= prob_val)
                #print(precision_, recall_)
                #auprc_ = auc(x= recall_, y= precision_)
                auprc_ = average_precision_score(y_score= prob_val, y_true= Yval)
            else:
                score_train = model.predict(Xtrain)[:, 1]
                score_val = model.predict(Xval)[:, 1]
                ###
                auc_ = roc_auc_score(y_score= score_val, y_true= Yval)
                
                ###precision_recall_area_under_the _curve
                #precision_, recall_, __ = precision_recall_curve(y_true= Yval, probas_pred= prob_val)
                #auprc_ = auc(x= recall_, y= precision_)
                auprc_ = average_precision_score(y_score= score_val, y_true= Yval)
                
            ####
            if risk_score:
                score_test = model.predict(Xtest).ravel()
                prob_test = IsoCal.calibrate(score_test)
                ###
                auc_test_ = roc_auc_score(y_score= prob_test, y_true= Ytest)
                
                ###precision_recall_area_under_the _curve
                #precision_, recall_, __ = precision_recall_curve(y_true= Ytest, probas_pred= prob_test)
                #auprc_test_ = auc(recall_, precision_)
                auprc_test_ = average_precision_score(y_score= prob_test, y_true= Ytest)
                
            else:
                score_test = model.predict(Xtest)[:, 1]
                
                #TMD_test = ThresholdMovingDecision(score_test, Ytest)
                auc_test_ = roc_auc_score(y_score= score_test, y_true= Ytest)
                ###precision_recall_area_under_the _curve
                #precision_, recall_, __ = precision_recall_curve(y_true= Ytest, probas_pred= prob_test)
                #auprc_test_ = auc(recall_, precision_)
                auprc_test_ = average_precision_score(y_score= score_test, y_true= Ytest)

                
            ####print all results
            print('Partial AUROC (validation):', np.round(auc_, 3))
            print('Partial AUROC (test):', np.round(auc_test_, 3))
            print('Partial AUPRC (validation)):', np.round(auprc_, 3))
            print('Partial AUPRC (test):', np.round(auprc_test_, 3))
            print('***** ***** ***** ***** *****')
            
            ######
            ###after fitting delete the training and validation variables
            del Xtrain
            del Xval
            del Ytrain
            del Yval
            
            
            ####save the results
            auc[fold] = auc_.round(3)
            auprc[fold] = np.round(auprc_, 3)
            auc_test[fold] = auc_test_.round(3)
            auprc_test[fold] = np.round(auprc_test_, 3)
            fold += 1
            #phi.append(phi_)
            
            #### RESET WEIGHTS
            model.set_weights(weights)
            
            ###CLEAR MODEL
            #K.clear_session()
            
            ###delete varibale model...
            #del model_fit
            
            print('pino')
            
        ######
        print('**** **** **** **** ****')
        print('AUROC (Validation):', np.mean(auc).round(2), np.round(sem(auc), 2))
        print('AUPRC (Validation):', np.mean(auprc).round(2), np.round(sem(auprc), 2))
        print("%%%%% ***** %%%%% ***** %%%%% ***** %%%%% *****")
        print('AUROC (Test):', np.mean(auc_test).round(2), np.round(sem(auc_test), 2))
        print('AUPRC (Test):', np.mean(auprc_test).round(2), np.round(sem(auprc_test), 2))
        
        #print('Phi:', np.mean(phi).round(2), sem(phi).round(2))
        
        ### ### ###
        jack_auroc = jackknife_stats(np.array(auc), np.mean)
        jack_auprc = jackknife_stats(np.array(auprc), np.mean)
        
        result = {'AUROC': jack_auroc[0].round(2),
                  'AUROC_err': np.round(jack_auroc[2], 2),
                  'AUPRC': jack_auprc[0].round(2),
                  'AUPRC_err': np.round(jack_auprc[2], 2),
                  'Test_Data': (Xtest, Ytest)}
        
        return result
    
    def fit_model(self, X, Y, model, cv= 5, rate_decay = 1e-7, 
                  batch_size= 32, epochs= 1000, verbose= 1, patience= 5, warm_start= False):
        
        if warm_start:
            model = PRETRAIN.make_pretraining(model)
    
        
        ###define Kfold-Cross-Validation
        kfold = KFold(5, shuffle= True, random_state= 58)
        
        ###
        PLATT = PlattCalibration()
        
        ###callbacks
        ###early_stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0, 
                                        patience= patience)
        
        ##lr_scheduler
        def LRScheduler(epoch, lr= model.optimizer.lr.numpy(), rate_decay= rate_decay):
            ### linear decreasing
            lrate = lr -rate_decay 
            return lrate
        lrate = tf.keras.callbacks.LearningRateScheduler(LRScheduler)
        
        ###metrics
        auc = []
        auc_test = []
        mcc = []
        mcc_test = []
        phi = []
        
        Xuse, Xtest, Yuse, Ytest = train_test_split(X, Y, train_size=.8, shuffle= True, random_state= 85) 
        
        for index_train, index_test in kfold.split(Xuse):
            
            ###train-test split
            Xtrain, Xval = Xuse[index_train], Xuse[index_test]
            Ytrain, Yval = Yuse[index_train], Yuse[index_test]
            
            Utest = np.hstack([Ytest.reshape(-1, 1), 1-Ytest.reshape(-1, 1)])
            
            ### call fit
            model.fit(Xtrain, Ytrain, 
                      validation_data = (Xval, Yval),
                      epochs= epochs, 
                      batch_size= batch_size, 
                      verbose= verbose,
                      callbacks=[lrate, early_stopping])
            
            #####
            score_train = model.predict(Xtrain)[:, 1]
            score_val = model.predict(Xval)[:, 1]
            #platt_coeffs = PLATT.get_coeffs(score_train, Ytrain)
            #PROB_VAL = PLATT.scaling(platt_coeffs, score_val)
            auc_ = roc_auc_score(y_score= score_val, y_true= Yval)
            mcc_ = f1_score(y_pred=score_val.astype(int), y_true= Yval)
            ####
            score_test = model.predict(Xtest)[:, 1]
            #platt_coeffs = PLATT.get_coeffs(score_train, Ytrain)
            #PROB_TEST = PLATT.scaling(platt_coeffs, score_test)
            auc_test_ = roc_auc_score(y_score= score_test, y_true= Ytest)
            mcc_test_ = matthews_corrcoef(y_pred=(score_test>.5).astype(int), y_true= Ytest)
            
            print('Partial AUC (validation):', np.round(auc_, 3))
            print('Partial AUC (test):', np.round(auc_test_, 3))
            print('Partial F1_score:', np.round(mcc_, 3))
            print('Partial F1_score (test):', np.round(mcc_test_, 3))
            print('***** ***** ***** ***** *****')
            #phi_ = matthews_corrcoef(y_pred= model.predict(Xtest), y_true= Ytest)
            
            auc.append(auc_)
            mcc.append(mcc_)
            auc_test.append(auc_test_)
            mcc_test.append(mcc_test_)
            
            #phi.append(phi_)
            
            #### RESET WEIGHTS
            #model.set_weights(weights)
            
            ###CLEAR MODEL
            #K.clear_session()
            
            ###delete varibale model...
            #del model_fit
            
        ######
        print('**** **** **** **** ****')
        print('AUC (Validation):', np.mean(auc).round(2), np.round(sem(auc), 2))
        print('F1_score (Validation):', np.mean(mcc).round(2), np.round(sem(mcc), 2))
        print('AUC (Test):', np.mean(auc_test).round(2), np.round(sem(auc_test), 2))
        print('F1_score (Test):', np.mean(mcc_test).round(2), np.round(sem(mcc_test), 2))
        
        #print('Phi:', np.mean(phi).round(2), sem(phi).round(2))
        
        ### ### ###
        result = {'AUC': np.mean(auc).round(2),
                  'AUC_err': np.round(sem(auc), 2),
                  'F1_score': np.mean(mcc).round(2),
                  'F1_score_err': np.round(sem(mcc), 2),
                  'Test_Data': (Xtest, Ytest),
                  'Train_Data': (Xtrain, Ytrain),
                  'model':model}
        
        return result
    
    
    def test(self, model_, Xval, Yval, Xtrain, Ytrain):
        
        ###
        PLATT = PlattCalibration()
        
        score_train = model.predict(Xtrain)
        score_val = model.predict(Xval)
        platt_coeffs = PLATT.get_coeffs(score_train, Ytrain)
        PROB_VAL = PLATT.scaling(platt_coeffs, score_val)
        auc_ = roc_auc_score(y_score= score_val, y_true= Yval)
        mcc_ = matthews_corrcoef(y_pred=(PROB_VAL>.5).astype(int), y_true= Yval)
        
        ######
        print('**** **** **** **** ****')
        print('AUC:', np.mean(auc_).round(2))
        print('MCC:', np.mean(mcc_).round(2))
        
        ######
    
##############
##############
##############
    
class DeconvFeatMap():
       
    def getfeat_maps(self, X, CNNnet):
        
        """
        ## A function to get all activation maps for the input X
        X == trial signal, tensor rank 2
        CNNnet --> the 1-D CNN net I want to use
        """
        ####
        trial_signal = np.expand_dims(X, axis= 0)
        pippo = [layer.output for layer in CNNnet.layers[1:]]
        memmo_model = tf.keras.models.Model(CNNnet.input, pippo)
        feat_map= memmo_model.predict(trial_signal, verbose= None)
        return feat_map
   

    def inverse_activation_function(self, X, kind= 'relu', inverse_rational_activation_function= None):

        "Make the inverse of the activation function"
        
        if kind=='relu':
            return np.maximum(0, X)
        if kind == 'softplus':
            arg_inv_softplus = np.maximum(1e-16, np.exp(X)-1)
            return np.log(arg_inv_softplus)
        if kind == 'sigmoid':
            return logit(X)
        if kind == 'tanh':
            eps= 1e-10
            Y = np.clip(X, a_min= -1+eps, a_max = 1-eps)
            return np.arctanh(Y)
        if kind == 'linear':
            return X
        if kind == "rational":
            X0 = X.copy()
            for kk in range(X.shape[1]):
                for ll in range(X.shape[2]):
                    x0 = inverse_rational_activation_function[kk, ll](X[:, kk, ll])
                    X0[:, kk, ll] = x0
            return X0
    
    def min_activation_function(self, kind= 'relu'):

        "Make the inverse of the activation function"
        
        if kind=='relu':
            return 0
        if kind == 'softplus':
            #arg_inv_softplus = np.maximum(1e-16, np.exp(X)-1)
            return 0
        if kind == 'sigmoid':
            return 0
        if kind == 'tanh':
            #eps= 1e-10
            #Y = np.clip(X, a_min= -1+eps, a_max = 1-eps)
            return -1
        if kind == 'linear':
            return -10
    
    def unpooling(self, Afeat, Bfeat, poolsize, actv, use_switch_variables= True):
        
        """
        Unpooling Layer. 
        Afeat --> lower level feature map  (After Maxpooling)
        Bfeat --> higher level feature map (Before Maxpooling)
        poolsize --> size of the pooling operator (CNN charateristic)
        """        
       
        ##unpool the Afeat with a zero-order spline
        X = np.dstack([np.repeat(Afeat[:, :, kk], poolsize) for kk in range(Afeat.shape[2])])        
            
        ###give this reconstruction the desired size
        Xpad = Bfeat.shape[1]-X.shape[1]

        if Xpad > 0:
            X = np.pad(X, pad_width=((0, 0), (0, Xpad), (0, 0)), mode= 'edge')
        elif Xpad<0:
            X = X[:, 0:Bfeat.shape[0], :]  
            
        if use_switch_variables:
            
            RES = Bfeat-X
            delta = 98e-2
            lambda_scale = npr.beta(.99, .01, size= RES.shape)
            X = X + RES#*lambda_scale
            
        else:
            RES = Bfeat-X
            RES == 0 ###so, the maxpooled actual values
            X = X*RES.astype(float) ### put all the others to zero

        return X
        
    
    def Inverse_RAFs(self, X, model_, rational_weights):
    
        """
        Deterimne the Inverse Ratiional Activation Functions
        X--> Training or sample data
        model_ --> 1-D CNN fitted model
        rational_weights --> a list containing the weights in a tensor shape
        """
    
        ###get the CONV and RAF layers
        index_conv = np.array([kk for kk in range(len(model_.layers)) if 'conv1d' in model_.layers[kk].name])
        index_actv = np.array([kk for kk in range(len(model_.layers)) if 'rational' in model_.layers[kk].name])

        ###
        CONVs = tf.keras.models.Model(model_.input, [model_.layers[ii].output for ii in index_conv])
        RAFs = tf.keras.models.Model(model_.input, [model_.layers[ii].output for ii in index_actv])

        ### find the sample size of the "interpolating population"
        Nindex = int(.48/((2e-2)**2+.48/X.shape[0]))
        index = npr.choice(np.arange(X.shape[0]), size= Nindex, replace= False)
        ### evaluation of raf and conv layers at the "interpolating population"
        xx = X[index]
        rafs = RAFs.predict(xx, verbose= None)
        convs= CONVs.predict(xx, verbose= None)

        ####TRIPLE for cycle to get the list with all the Inverse RAFs
        Inverse_RAFs = []
        RF = RationalFit()
        for ii in range(len(rafs)):
            #PHI = np.empty((rafs[ii].shape[1], rafs[ii].shape[2]), dtype= 'O')
            PHI = []
            for jj in range(rafs[ii].shape[1]):
                for kk in range(rafs[ii].shape[2]):
                    ##get the particula coefficients
                    RF_weights = rational_weights[ii]
                    ALPHA_c = RF_weights[0:4, jj, kk]
                    BETA_c = RF_weights[4:7, jj, kk]
                    ###make the interpolation (define the inverse function)
                    #xxx, xxx_where = np.unique(rafs[ii][:, jj, kk], return_index=True)
                    xxx = np.linspace(-10, 10, 1000)
                    ul, ur = RF.rational_function(np.array([-100, 100]), ALPHA_c, BETA_c)
                    phi = interp1d(RF.rational_function(xxx, ALPHA_c, BETA_c), 
                                   xxx,
                                   kind= 'zero', 
                                   bounds_error=False, 
                                   fill_value='extrapolate')
                    PHI.append(phi)
                    #PHI[jj, kk] = phi
            Inverse_RAFs.append(np.array(PHI).reshape(rafs[ii].shape[1], rafs[ii].shape[2]))

        #######
        # RETURN a list of arrays. Each element of these arrays contains a spline (inverse RAF) for each feature at each spatial location 

        return Inverse_RAFs
        
    
    #def unpooling(self, Afeat, Bfeat, poolsize, actv= 'tanh', use_switch_variables= None):

    #    """
    #    Unpooling Layer. 
    #    Afeat --> lower level feature map  (After Maxpooling)
    #    Bfeat --> higher level feature map (Before Maxpooling)
    #    poolsize --> size of the pooling operator (CNN charateristic)
    #    """
        
        ##########OLD METHOD###########
        ##look at the padding
    #    DELTA_PAD = poolsize-Bfeat.size%poolsize
        
        ## make padding
    #    X = np.pad(Bfeat, pad_width= (0, DELTA_PAD), mode= 'edge')
        
        ### look at where the unpooling must be executed
    #    index_rest = X.reshape(-1, poolsize).argmax(axis= 1)
    #    index = poolsize*np.arange(0, index_rest.shape[0], 1) + index_rest
    #    indexmin = min(index.size, Afeat.size)

        #MAKE THE UNPOOLING (Switch Variable)
    #    Ndiff = index.shape[0]-Afeat.shape[0]
    #    if Ndiff >0:
    #        Afeat= np.pad(Afeat, pad_width=(0, Ndiff), mode= 'constant')
    #    else:
    #        Afeat= Afeat[0:index.shape[0]]

        
    #    unpooled = np.zeros(X.size)
    #    min_actv = self.min_activation_function(actv)
    #    for ii in range(X.size):
            
    #        if ii in index:
    #            unpooled[ii] = X[ii]
    #        else:
    #            A0 = np.argmin(np.abs(index-ii))
    #            A1 = A0+1
                
     #           unpooled[ii] = npr.uniform(min_actv, np.minimum(X[A0], X[A1]), size= 1)


     #   return unpooled
    
    
    #def unpooling(self, Afeat, Bfeat, poolsize= 2, actv= 'relu', switch_variables= False, use_PME= False):

    #    """
    #    Unpooling Layer. 
    #    Afeat --> lower level feature map  (After Maxpooling)
    #    Bfeat --> higher level feature map (Before Maxpooling)
    #    poolsize --> size of the pooling operator (CNN charateristic)
    #    """
        
    #    ##look at the padding
    #    DELTA_PAD = Bfeat.size%poolsize        
        
    #    if switch_variables:
    #        X = np.pad(Bfeat, pad_width= (0, DELTA_PAD), mode= 'edge')
    #        Agrad = np.gradient(Afeat)
            
            ###split X per pooling domain
    #        splitX = np.split(X, int(X.size/poolsize))

            ###repelace the minimal values (per chunk) with 0
    #        splitX_array = [np.piecewise(kk, [kk< np.max(kk)], funclist= [np.max(kk)-1, np.max(kk)]) for kk in splitX]

            ####impute the variables that are discarded in the switch variable method
    #        box = []
    #        count_error = 0
    #        index = 0
    #        for item in splitX_array:
                
    #            censored_value = np.max(item)-1
    #            if (item == censored_value).sum() > 0:
    #                if use_PME:
    #                    try:
    #                        pme = PME_retirvial_distribution(rho=Agrad[index] , 
    #                                                         xmax= item.max(), 
    #                                                         xmin= self.min_activation_function(actv)) 
    #                        item[item == censored_value] = pme.generate_data((item == censored_value).sum())
    #                    except:
    #                        item[item == censored_value] = np.max(item)*npr.random((item == censored_value).sum())
    #                        count_error += 1
    #                else:
    #                    item[item == censored_value] = npr.uniform(self.min_activation_function(actv), 
    #                                                               np.max(item), 
    #                                                               (item == censored_value).sum())
    #            box.append(item)
    #            index += 1

    #        #print(count_error/len(splitX_array))
    #        return np.concatenate(box)
        
    #    else:
            ###use the linear interpolation
    #        Z = np.linspace(0, 1, Afeat.size)
    #        PHI = interp1d(Z, Afeat, kind= 'linear', bounds_error=False, fill_value=(Afeat[0], Afeat[-1]))
    #        newZ = np.linspace(0, 1, Bfeat.size)
    #        return newZ
        
        ###################################################
        ###Implement Switch Variables
        
        ###make an empty array with the same shape of Bfeat
        #DELTA_PAD = poolsize-Bfeat.size%poolsize
        
        ## make padding -- Create a vector with the same values of Bfeat
        #Bfeat_T = np.pad(Bfeat, pad_width= (0, DELTA_PAD), mode= 'edge')
        
        ##find the location where the maxima occur
        #max_loc = Bfeat_T.reshape(-1, poolsize).argmax(axis= 1)
        
        ###reset Bfeat (i.e., Bfeat_T is now all made of nans)
        #Bfeat_T = np.repeat(np.nan, Bfeat_T.size).reshape(-1, poolsize)
        
        ##define Afeat_T
        #delta_pad = Bfeat_T.shape[0]-Afeat.shape[0]
        #if delta_pad >0:
        #    Afeat_T = np.pad(Afeat, pad_width=(0, delta_pad), mode='edge')
        #else :
        #    Afeat_T = Afeat[0:Bfeat_T.shape[0]]
        
        
        ##get the index where the maximum occurs
        #index_max = np.vstack([np.arange(Bfeat_T.shape[0]), max_loc]).T
        
        ###Allocate the maximum values via switch variables
        #for item in index_max:
        #    ii, jj = item
        #    Bfeat_T[ii, jj] = Afeat_T[ii]
            
        ####make a primitive version of the unpooled map
        #unpooled = Bfeat_T.ravel()
        
        ###########################################
        ######## Interpolation of the leftovers ###
        ###########################################
        
        ###prepare the evaluation of the 2nd derivative
        #Bfeat_spline = UnivariateSpline(np.arange(Bfeat.size), Bfeat, k=5)
        #Bfeat_dd = Bfeat_spline.derivative(n=2)(np.arange(Bfeat.size))
        
        ##take the locations of the "leftovers"
        #unpooled_nan =np.where(np.isnan(unpooled))[0]
        
        ##take the locations of the switch variables
        #unpooled_NOnan = np.where(~np.isnan(unpooled))[0]
        
        
        #print('************')
        #print(unpooled.shape, unpooled_nan.shape, unpooled_NOnan.shape)
        #print('************')
        
        
        ###Now make the interolation
        #for index in unpooled_nan:
            
            ###get the closest switch variables to the leftover under consideration
        #    A0 = np.argmin(np.abs(unpooled_NOnan-index))
        #    A1 = A0+1
            ### make interpolation
        #    unpooled[item] = .5*(unpooled[A0]+unpooled[A1]-Bfeat_dd[item]*(A1-A0)**2)
            
        
        #return unpooled
    

    ### ### ### ### ### ###
    def invert_one_layer(self, 
                         feat_map_before,
                         feat_map_after, 
                         feat_map_conv,
                         weights, 
                         actv,
                         Bias = 0,
                         strides = 1,
                         poolsize = 2, 
                         iraf= None,
                         use_switch_variables= True):

        """
        invert one single composition of CONV+ACTV+MAXPOOL
        
        ##################################
        ### feat_map_before --> feature map at higher level (before maxpooling)
        ### feat_map_after --> feature map at lower level (after maxpooling)
        ### feat_map_conv --> feature map just after the convolutions
        ### weights --> weights of the CNN
        
        #########
        ffnn --> whether to use the feed-forwad activation function
        weights_ffnn --> weights of Dense layers when the FFNN activation is used
        feat_map_ffnn --> feat maps evalauted after each Dense Layer of FFNN
        """
        
        # consider dimension of feat map before the pooling operator
        _, timesize, featsize = feat_map_before.shape    

        # define reconstructed data
        rec_inst = np.zeros((feat_map_before.shape[0], strides*feat_map_before.shape[1], feat_map_before.shape[2]))
        
        ####
        #for kfeat in range(0, featsize):
            
            ### 1. UNPOOLING
            #### USE SWITCH VARS for unpooling
            ## make unpooling
        #    zz = self.unpooling(feat_map_after[:, :, kfeat].ravel(), 
        #                        feat_map_before[:, :, kfeat].ravel(), 
        #                        poolsize= poolsize, 
        #                        actv= actv)
            
            #zz= feat_map_before[:, :, kfeat].ravel() ###SUPPOSE MAX-POOLING INVERSION IS OPTIMAL!
    
        ### 1. UNPOOLING
        zz = self.unpooling(feat_map_after, Bfeat= feat_map_before, poolsize= poolsize, 
                            actv= actv, 
                            use_switch_variables = use_switch_variables)
        
        ###DO NOT RUN. USE ONLY DURING TEST MODE
        #zz = feat_map_before.copy()
        
        ###2. INVERT the activation function 
        zz = self.inverse_activation_function(zz, 
                                              kind = actv, 
                                              inverse_rational_activation_function= iraf)


        
            ####
            ##3. Anti-Stride (approximation via spline order 0)
        #    if strides > 1:
        #        pino = np.repeat(zz, strides)
                #kernel_size = 1+2*int(strides/2) 
                #window = boxcar(kernel_size)/(kernel_size)
                #Npad =  pino.shape[0]-kernel_size+1
                #padr = int(Npad/2)
                #padl = Npad-padr
                #pino = np.pad(pino, pad_width= (padl, padr), mode= 'edge')
                #pino = np.convolve(pino, window, mode= 'valid')
        #        rec_inst[:, :, kfeat] = pino.reshape((rec_inst.shape[0], rec_inst.shape[1]))
        #    else:
        #        rec_inst[:, :, kfeat] = zz

        ###############
        ## ALTERNATIVE -- DO NOT RUN!!!
        ###############
        ## 1. UNPOOLING
        #UNPOOLING = tf.keras.Sequential()
        #UNPOOLING.add(tf.keras.layers.UpSampling1D(size= poolsize))
        ##
        #rec_inst = UNPOOLING.predict(feat_map_after)
        #Delta = feat_map_before.shape[1]-rec_inst.shape[1]
        #rec_inst = np.pad(rec_inst, pad_width=((0, 0), (int(Delta*.5), int(Delta*.5)+Delta%2), (0, 0)), mode='constant')
        
        
        ### 2.Deactivation
        #rec_inst = self.inverse_activation_function(X = rec_inst, kind = actv)
        #######################
                      
        ###################
        ### 3. Transpose Convlution
        ##define a keras model to make it!
        DECONV = tf.keras.Sequential()
        DECONV.add(tf.keras.layers.Conv1DTranspose(filters = weights.shape[1], 
                                                       kernel_size = weights.shape[0], 
                                                       strides=1, 
                                                       padding='valid', 
                                                       use_bias=False, 
                                                       input_shape = (zz.shape[1], zz.shape[2])))


        DECONV.set_weights([weights])
        

        ### do transpose_conv
        transpose_dec = DECONV.predict(zz, verbose = None)
            
        return transpose_dec 
    
    
    ### ### ### ### ### ###
    def invert_block_ffnn(self, 
                         feat_map_after,
                         feat_map_before,
                         feat_map_dense_2,
                         feat_map_actv_,
                         feat_map_dense_1,
                         feat_map_conv,
                         weights_conv, 
                         weights_ffnn_1,
                         weights_ffnn_2,
                         strides = 1,
                         poolsize = 2, 
                         use_switch_variables= True):

        """
        invert one single composition of CONV+FFNN_ACTV+MAXPOOL
        
        ##################################
        feat_map_after --> feat. map after maxpooling
        feat_map_before --> feat. map before maxpooling
        feat_map_dense_2 --> feat. map 2nd dense layer
        feat_map_actv_ --> feat. activation in between the two dense layer
        feat_map_dense_1 --> feat. map 1st dense layer
        feat_map_conv --> convolutiional layer
                         
        
        #########
        ffnn --> whether to use the feed-forwad activation function
        weights_conv --> weights convolution
        weights_ffnn --> weights of Dense layers when the FFNN activation is used
        
        """
        
        # consider dimension of feat map before the pooling operator
        _, timesize, featsize = feat_map_before.shape    

        # define reconstructed data
        rec_inst = np.zeros((feat_map_before.shape[0], strides*feat_map_before.shape[1], feat_map_before.shape[2]))
        
        ### 1. UNPOOLING
        zz = self.unpooling(feat_map_after, Bfeat= feat_map_before, poolsize= poolsize, 
                            actv= None, 
                            use_switch_variables = use_switch_variables)
        
        ###DO NOT RUN. USE ONLY DURING TEST MODE
        #zz = feat_map_before.copy()
        
        ###2. INVERT the FFNN activation function 
        
        
        ###2.1 invert the last activation function
        zz = self.inverse_activation_function(zz, kind = 'tanh', 
                                              inverse_rational_activation_function= False)
        ###2.2 invert the last dense layer
        winv = np.linalg.inv(weights_ffnn_2)
        zz = np.matmul(zz, winv)
        
        ###2.3 invert the middle activation function
        zz = self.inverse_activation_function(zz, kind = 'relu', 
                                              inverse_rational_activation_function= False)
        
        ###2.4 invert the last dense layer
        winv = np.linalg.inv(weights_ffnn_1)
        zz = np.matmul(zz, winv)
                      
        ###################
        ### 3. Transpose Convlution
        ##define a keras model to make it!
        DECONV = tf.keras.Sequential()
        DECONV.add(tf.keras.layers.Conv1DTranspose(filters = weights_conv.shape[1], 
                                                       kernel_size = weights_conv.shape[0], 
                                                       strides=1, 
                                                       padding='valid', 
                                                       use_bias=False, 
                                                       input_shape = (zz.shape[1], zz.shape[2])))


        DECONV.set_weights([weights_conv])
        

        ### do transpose_conv
        transpose_dec = DECONV.predict(zz, verbose = None)
            
        return transpose_dec 
    
    
    def deconv(self, 
               feat_map_before, 
               feat_map_after, 
               feat_map_conv, 
               weights, 
               bias = 0, 
               strides= 1, 
               poolsize= 2, 
               actv = 'relu', 
               IRAF = None,        
               use_switch_variables = True):

        """
        ##################################
        feat_map_after --> feat. map after maxpooling
        feat_map_before --> feat. map before maxpooling
        feat_map_dense_2 --> feat. map 2nd dense layer
        feat_map_actv_ --> feat. activation in between the two dense layer
        feat_map_dense_1 --> feat. map 1st dense layer
        feat_map_conv --> convolutiional layer 
        ####
        ffnn --> whether to use the feed-forwad activation function
        weights_conv --> weights convolution
        weights_ffnn --> weights of Dense layers when the FFNN activation is used
        
        Make the deconvolution!
        """
        
        ###preparation
        deconv_hidden = []
        one_deconv = feat_map_after[-1]
        
        ####define IRAF when no RAF is used
        if actv != 'rational':
            IRAF = [None for ii in range(len(feat_map_before))]
        
        ####make deconvolution
        for kk in range(len(feat_map_before)-1, -1, -1):
            
            
            one_deconv = self.invert_one_layer(feat_map_before[kk],
                                                 one_deconv, 
                                                 feat_map_conv[kk],
                                                 weights[kk], 
                                                 actv,
                                                 bias,
                                                 strides,
                                                 poolsize, 
                                                 IRAF[kk], 
                                                 use_switch_variables = use_switch_variables)


            #deconv_hidden.append(one_deconv)


        ####make the Deconvolution Normalized 
        ###
        #normalized_map = Normalizer(norm='l1').fit_transform(one_deconv[0])
        #np.abs(normalized_map), one_deconv, deconv_hidden

        #return one_deconv, deconv_hidden
        return one_deconv
    
    
    def deconv_ffnn(self, 
                   feat_map_after,
                   feat_map_before,
                   feat_map_dense_2,
                   feat_map_actv_,
                   feat_map_dense_1,
                   feat_map_conv,
                   weights_conv, 
                   weights_ffnn_1,
                   weights_ffnn_2, 
                   strides= 1, 
                   poolsize= 2, 
                   use_switch_variables = True):

        """
        ##################################
        ### feat_map_before --> feature map at higher level (before maxpooling)
        ### feat_map_after --> feature map at lower level (after maxpooling)
        ### feat_map_conv --> feature map just after the convolutions
        ### weights --> weights of the CNN
        
        Make the deconvolution!
        """
        
        ###preparation
        deconv_hidden = []
        one_deconv = feat_map_after[-1]
        
        ####make deconvolution
        for kk in range(len(feat_map_before)-1, -1, -1):
            
            
            one_deconv = self.invert_block_ffnn(feat_map_after[kk],
                                             feat_map_before[kk],
                                             feat_map_dense_2[kk],
                                             feat_map_actv_[kk],
                                             feat_map_dense_1[kk],
                                             feat_map_conv[kk],
                                             weights_conv[kk], 
                                             weights_ffnn_1[kk],
                                             weights_ffnn_2[kk],
                                             strides = 1,
                                             poolsize = 2, 
                                             use_switch_variables= True)


        return one_deconv
    
        
    def get_maps_per_block(self, X, CNNnet, activation= 'relu'):
        
        """
        ...
        
        Reutrn in order:
        1. A list with the convolution maps (maps just after the convolution)
        2. A list with the activation maps (maps just after the activation function, just before the maxpooling)
        3. A list with the pooled maps (maps after the pooling operator)
        """
        
        ##get the layers' index
        index_conv = np.array([kk for kk in range(len(CNNnet.layers)) if 'conv1d' in CNNnet.layers[kk].name])
        #
        if activation=='rational':
            index_actv = np.array([kk for kk in range(len(CNNnet.layers)) if 'rational' in CNNnet.layers[kk].name])
        elif activation == 'FFNN':
            index_dense = np.array([kk for kk in range(len(CNNnet.layers)) if 'dense' in CNNnet.layers[kk].name])
            index_actv = np.array([kk for kk in range(len(CNNnet.layers)) if 'activation' in CNNnet.layers[kk].name])
        else:
            index_actv = np.array([kk for kk in range(len(CNNnet.layers)) if 'activation' in CNNnet.layers[kk].name])
        #
        index_maxp = np.array([kk for kk in range(len(CNNnet.layers)) if 'max_pooling' in CNNnet.layers[kk].name])
        
        ###get all featmaps
        maps = self.getfeat_maps(X = X, CNNnet= CNNnet)
        
        ###
        Cmax = len(index_conv)
        list_conv = [maps[index_conv[kk]-1] for kk in range(Cmax)]
        list_actv = [maps[index_actv[kk]-1] for kk in range(Cmax)]
        list_maxp = [maps[index_maxp[kk]-1] for kk in range(Cmax)]
        list_weights = [item.numpy() for item in CNNnet.weights if 'kernel' in item.name]
        
        if activation == 'FFNN':
            list_dense = [maps[index_dense[kk]-1] for kk in range(2*Cmax)]
            list_actv = [maps[index_actv[kk]-1] for kk in range(2*Cmax)]
            list_weights_conv = [item.numpy() for item in CNNnet.weights if 'conv1d' in item.name]
            list_weights_dense = [item.numpy() for item in CNNnet.weights if 'dense' in item.name]
            return list_conv, list_actv, list_maxp,  list_dense, list_weights_conv, list_weights_dense
        else :
            return list_conv, list_actv, list_maxp, list_weights  
    
    def relevance_features(self, X, Y, CNNmodel, kind= 'average'):
    
        """
        Evaluate Relevence of each Time-Series feature via a feature inspection-based approach.
        The model must return the probability of each class (e.g., via Softmax)

        ******
        X --> Test Data
        Y --> Test Data
        model --> trained or cross-validated model
        """
        
        if kind == 'average':
            Nfeats = X.shape[2]
            #phi0 = matthews_corrcoef(y_pred= np.argmax(CNNmodel.predict(X), axis= -1), 
            #                         y_true= Y)

            auc0 = roc_auc_score(y_score=CNNmodel.predict(X, verbose= None).ravel(), y_true= Y)

            PHI = []
            for jj in range(Nfeats):

                ###At each iteration make one feature silent
                Xsilent = X.copy()
                Xsilent[:, :, jj] = 0

                ###Evaluate AUC
                #class_pred = np.argmax(CNNmodel.predict(Xsilent), axis= -1)
                #phi_jj = matthews_corrcoef(y_pred=class_pred, y_true= Y)
                auc_jj = roc_auc_score(y_score=CNNmodel.predict(Xsilent, verbose= None).ravel(), y_true= Y)

                performance = np.minimum(0, auc_jj-auc0)
                PHI.append(performance)

            ###make a rank statistic
            #ranks = rankdata(np.array(PHI), method='average')
            #ranks = ranks/ranks.sum()
            #ranks = softmax(PHI)
            ranks = np.array(PHI)/auc0 
            return ranks
        
        elif kind == 'micro':
            
            Nfeats = X.shape[2]
            score0 = CNNmodel.predict(X)*Y+CNNmodel.predict(X)*(1-Y)
            
            Silent_Score = []
            for jj in range(Nfeats):

                ###At each iteration make one feature silent
                Xsilent = X.copy()
                Xsilent[:, :, jj] = 0

                ###Evaluate AUC
                #class_pred = np.argmax(CNNmodel.predict(Xsilent), axis= -1)
                #phi_jj = matthews_corrcoef(y_pred=class_pred, y_true= Y)
                score_ = CNNmodel.predict(Xsilent)*Y+CNNmodel.predict(Xsilent)*(1-Y)

                performance = (score_-score0).ravel()
                Silent_Score.append(performance)
            
            Silent_Score = np.array(Silent_Score)
            #ranks = softmax(Silent_Score, axis= 0)
            ranks = .5*(Silent_Score+1)
            return ranks.T
        
        else:
            print('MOOOOOOO!')
            return 
            
    
    def Reconstruction(self, X, CNNmodel, activation= 'relu', use_switch_variables= True):
    
        """
        Reconstruct the instances given the model

        X--> Instances
        """

        ###make reconstrunction
        Recs = []
        
        DI = dt.now()
        
        ###if activation == rational, then get the inverse rational functions
        if activation == 'rational':
            
            ###select the rational parameters
            Fw = [item for item in CNNmodel.get_weights() if len(item.shape) == 3]
            Lfw = len(Fw)
            
            ###list containing the rational weights (in a tensor shape: weight_typeXTime-DomainXlatent_Feature)
            rational_weights = [np.array(list(np.split(np.array([cube.numpy() for cube in CNNmodel.weights if 'rational' in cube.name]), Lfw)[pp])) for pp in range(Lfw)]
            IRAF = self.Inverse_RAFs(X, CNNmodel, rational_weights)
        else:
            IRAF = None
        
        
        count = 0
        
        for item in X:
            count += 1
            ###get the single instance
            #item = np.expand_dims(item, axis= 0)
            
            if activation=='FFNN':
                
                
                Fconv, Factv, Fpool, Fdense, Fw, Fwd = pino.get_maps_per_block(item, 
                                                                               CNNmodel, 
                                                                               activation='FFNN')
                #### #### ####
                Factv_1 = [Factv[2*kk] for kk in range(len(Fconv))]
                Factv_2 = [Factv[2*kk+1] for kk in range(len(Fconv))]
                #
                Fdense_1 = [Fdense[2*kk] for kk in range(len(Fconv))]
                Fdense_2 = [Fdense[2*kk+1] for kk in range(len(Fconv))]
                #
                Fweight_1 = [Fwd[2*kk] for kk in range(len(Fconv))]
                Fweight_2 = [Fwd[2*kk+1] for kk in range(len(Fconv))]
                
                ###
                deconvs = self.deconv_ffnn(feat_map_after = Fpool,
                                           feat_map_before = Factv_2,
                                           feat_map_dense_2 = Fdense_2,
                                           feat_map_actv_ = Factv_1,
                                           feat_map_dense_1 = Fdense_1,
                                           feat_map_conv = Fconv,
                                           weights_conv = Fw, 
                                           weights_ffnn_1 = Fweight_1,
                                           weights_ffnn_2 = Fweight_2, 
                                           strides= 1, 
                                           poolsize= 2, 
                                           use_switch_variables = True)
                
                ####
                Recs.append(deconvs)
                DF = dt.now()
                
            
            else:
                
                Fconv, Factv, Fpool, Fw = self.get_maps_per_block(item, CNNmodel, activation)
            
                ###if activation == rational, then discard the coefficients of RAF
                if activation == 'rational':
                    Fw = [item for item in Fw if len(item.shape) == 3]

            
                ###get the deconvolutions
                deconvs = self.deconv(feat_map_before=Factv, 
                                         feat_map_after=Fpool, 
                                         feat_map_conv=Fconv, 
                                         weights= Fw, 
                                         actv=activation, 
                                         IRAF = IRAF, 
                                         use_switch_variables = use_switch_variables)

                Recs.append(deconvs)
                DF = dt.now()

        ####
        recs_ = np.vstack(Recs)
        
        ###make std unitary
        #if unitary_std:
        #    for kk in range(recs_.shape[0]):
        #        recs_[kk] = StandardScaler().fit_transform(recs_[kk])
        
        print(DF-DI)
        
        return recs_
        
    
    
    def SaliencyMap(self, X, Recs, CNNmodel= None, Y= None, multi_channel= False, 
                    activation= 'relu', Time_Scale= .125):
    
        """
        Determine the SaliencyMap given the model

        X--> Instances
        Recs --> Reconstructed instances 
        CNNmodel --> CNN model used to obtained the Reconstructed Instances via DNN. NO NEED to be specified if multi_channel= True 
        Y --> labels associated with the instances (NEED to be specified if multi_channel= True)
        multi_channel --> Boolean. True imples that a saliency map is evaluated per each time-series feature
        activation --> activation function used in CNN model. Default 'relu'
        
        """

        def __binarizer__(x, sigma= 1e-3):
            return 1-np.exp(-(np.std(x**2)/sigma)**2)
        
        def __sliding_window__(X, w_size, tau):
    
            """Chunk Time-Series slices after sliding a one-valued window
                X --> Time-Series (1-D)
                w_size --> sliding window size 
                tau --> stride of sliding window
            """


            Y = X.copy()
            Nmax = max(1, 1+int((Y.size-w_size)/tau))
            Z = np.empty((Nmax, w_size))
            for kk in range(0, Nmax):
                liscio_flag = kk*tau 
                Z[kk] = Y[liscio_flag:liscio_flag+w_size]
            return Z

        
        
        def __saliency__(x, y, W):
            
            """
            Internal function to evaluate salinecy (linear correlation between both reconstructed and true instance)
            
            x--> reconstructed instance
            y--> true instance
            """
            
            W0 = np.maximum(1, W)
            N = f.shape[0]
            K = N-W0+1
            
            ###remove artifacts from the reconstruted instance (highpass of frequency 1/W0)
            #window = butter(N= 3, Wn= 1/W0, btype= 'highpass')
            #x = filtfilt(window[0], window[1], x)
            
            ###evaluate the saliency coeff. (pearson)
            #F0, G0 = __sliding_window__(x, w_size=W0, tau= 1), __sliding_window__(y, w_size=W0, tau= 1)
            #score = np.array([pearsonr(F0[ii], G0[ii])[0]*__binarizer__(G0[ii]) for ii in range(F0.shape[0])])
            
            ###evaluate the saliency coeff. (chi^2)
            # Standardize
            F0 = StandardScaler(with_mean=True, with_std=True).fit_transform(x.reshape(-1, 1)).ravel()
            G0 = StandardScaler(with_mean=True, with_std=True).fit_transform(y.reshape(-1, 1)).ravel()
            ####
            Epsilon = (F0-G0)**2
            score = chi2.sf(Epsilon, df= 1)
            
            ####
            score = np.nan_to_num(score, nan= 0)
            
            #### #### ####Re-Adjust padding
            Res = N-score.size
            score = np.pad(score, pad_width=(int(Res*.5), int(Res*.5)+Res%2), 
                           mode= 'constant', constant_values= 0)
            #### #### #### 
            window_tukey = tukey(M= N, alpha= 5e-2)
            return np.array(score)*window_tukey
        
        
        SA = []
        for kk in range(X.shape[0]):
            
            ##### sorry for the double for!
            ### use Pearson's coeff. to evaluate the saliency
            sa = []
            for ii in range(X.shape[2]):
                f, g = Recs[kk, :, ii], X[kk, :, ii]
                #N = f.shape[0]
                #W = int(N*Time_Scale)
                #score_ = __saliency__(f, g, W)
                score_ = chi2.sf((f-g)**2, df= 1)
                sa.append(score_)
            
            ###
            SA.append(np.vstack(sa).T)
        
        #####
        SA = np.array(SA)
        
        if multi_channel == True:
            rel = self.relevance_features(X, Y, CNNmodel, kind='average')
            
            if rel.sum() ==0:
                rel = np.ones(rel.shape[0])
                
            one_channel_SA = np.array([np.average(item, axis=1, weights=rel) for item in SA])
            #window_sa = hamming(int(5e-2*one_channel_SA.shape[1]))
            #window_sa = window_sa/window_sa.sum()
            #one_channel_SA = np.apply_along_axis(np.convolve, axis= 1, 
            #                                     arr= one_channel_SA,
            #                                     v= window_sa, mode='same')
            return one_channel_SA
        else:
            
            #window_sa = hamming(int(5e-2*SA.shape[1]))
            #window_sa = window_sa/window_sa.sum()
            #multi_channel_sa = np.apply_along_axis(np.convolve, axis= 1, 
            #                           arr= SA, 
            #                           v= window_sa, 
            #                           mode= 'same')
            return SA
            
            
            
class Tools:
    
    def __init__(self):
        
        self.moo = 'MOO!'
        return 
        
        
    def naive_convolve(self, X, window):
    
        W = window.size
        N = X.size

        Wleft= int(W/2)
        Wright= W-Wleft-1
        Y = np.pad(X, pad_width=(Wleft, Wright), mode= 'reflect')

        return np.convolve(Y, window, 'valid')    
    
    
    def standardize_ts(self, TS, with_mean= True, with_std= True):
        
        """Make zero-mean and unitary SD a batch of TS
        
        TS --> my batch of TS
        
        """
        
        Zts = np.array([StandardScaler(with_mean= with_mean, with_std= with_std).fit_transform(item) for item in TS])
        return Zts
        
    def PowerSpectrum(self, X):
        
        ft = npf.fft(X)
        ps = np.abs(ft)**2
        freq = npf.fftfreq(d=1, n= X.size)
        ####
        plt.plot(freq[0:int(.5*ft.size)], ft[0:int(.5*ft.size)])
        plt.xlabel('Freq/(2*Nyquist_freq)')
        plt.ylabel('Power Spectrum Density')
        plt.grid(True)
        plt.show()
        return
        
    def ROAR(self, X, Y, S, model, metric= 'AUROC'):
    
        """
        Make a plot with the ROAR score.
        
        X --> Instances (nsamples X features)
        S --> Correspinding Saliency Maps
        model --> ANN trained model
        
        ### ### ###
        METRIC IMPLEMENTED : AUC
        otherwise
        MCC
        
        #### ROAR IMPLEMENTATION
        
        1. DIVIDE DATA IN 10 FOLDS and evalaute ROAR; comapre the results with 10 different random occusions
        
        """
        
        ### percetage of salient pixels
        N_sal_pxl = 11
        imp = np.linspace(0, 1, N_sal_pxl, endpoint=True)
        
        ###
        ROAR_metric= []
        ROAR_err= []
        #
        RANDOM_metric= []
        RANDOM_err= []
        
        
        ###number of Time-points
        Ntime = X.shape[1]
        
        ###
        Kroar = KFold(n_splits= 10, shuffle= True, random_state= 56)
        #Kroar = LeaveOneOut()
        
        for fold in range(N_sal_pxl):
            
            internal_metric = []
            internal_metric_rnd = []
            
            for index_keep, index_use in Kroar.split(X):

                ##select the data and the saliency maps
                Xuse = X[index_keep]
                Yuse = Y[index_keep]
                Suse = S[index_keep]

                ### saliency threshold
                q0 = 1-imp[fold]
                thr = np.quantile(Suse, q=q0)
                
                #if q0 == 0:
                #    thr = np.zeros(thr.shape)
                #elif q0 == 1:
                #    thr= np.ones(thr.shape)
                    
                ###take a value larger than "threshold-saliency", i.e., any value lower than threshold
                mask = (Suse>=thr).astype(int)
                Xsalient = Xuse*mask            
                
                #####
                #print(Xsalient.shape, Xsalient_at_random.shape, Yuse.shape)
                
                ###USE AUC METRIC
                if metric == 'AUROC':

                    ##salient 
                    Msal = roc_auc_score(y_score=model.predict(Xsalient, verbose= None), y_true= Yuse)
                    internal_metric.append(Msal)

                    ##random 
                    mrnd= []
                    for ll in range(10):
                        ####
                        mask_rnd = npr.choice([0, 1], size= Xuse.shape, p= [1-imp[fold], imp[fold]])
                        Xsalient_at_random = Xuse*mask_rnd
                        #####
                        Mrnd = roc_auc_score(y_score=model.predict(Xsalient_at_random, verbose= None), y_true= Yuse)
                        mrnd.append(Mrnd)
                    internal_metric_rnd.append(np.mean(Mrnd))
                
                elif metric == 'AUPRC':

                    ##salient 
                    Msal = roc_auc_score(y_score=model.predict(Xsalient, verbose= None), y_true= Yuse)
                    internal_metric.append(Msal)

                    ##random 
                    mrnd= []
                    for ll in range(10):
                        ####
                        rnd_tokens = npr.choice([0, 1], size=Xuse.shape[1], p= [1-imp[fold], imp[fold]])
                        mask_rnd = npr.choice([0, 1], size= Xuse.shape, p= [1-imp[fold], imp[fold]])
                        Xsalient_at_random = Xuse*mask_rnd
                        #####
                        Mrnd = average_precision_score(y_score=model.predict(Xsalient_at_random, verbose= None), y_true= Yuse)
                        mrnd.append(Mrnd)
                    internal_metric_rnd.append(np.mean(Mrnd))
                
                
                elif metric =='MCC':
                    
                    ##salient 
                    Msal = matthews_corrcoef(y_pred=(model.predict(Xsalient, verbose= None)>=.5).astype(int), y_true= Yuse)
                    internal_metric.append(Msal)

                    ##random 
                    mrnd= []
                    for ll in range(10):
                        ####
                        rnd_tokens = npr.choice([0, 1], size=Xuse.shape[1], p= [1-imp[fold], imp[fold]])
                        mask_rnd = npr.choice([0, 1], size= Xuse.shape, p= [1-imp[fold], imp[fold]])
                        Xsalient_at_random = Xuse*mask_rnd
                        #####
                        Mrnd = matthews_corrcoef(y_pred=(model.predict(Xsalient_at_random, verbose= None)>=.5).astype(int), 
                                                 y_true= Yuse)
                        mrnd.append(Mrnd)
                    internal_metric_rnd.append(np.mean(Mrnd))
                    
            #####Save Results
            ROAR_metric.append(np.mean(internal_metric))
            ROAR_err.append(sem(internal_metric))
            ###
            RANDOM_metric.append(np.mean(internal_metric_rnd))
            RANDOM_err.append(sem(internal_metric_rnd))
            ######
        
        #############
        #############
        ### MAKE PLOT
        plt.figure(1)
        ###make ROAR first
        init_, final_ = 360*48/90, 360*90/90
        kolors = sns.diverging_palette(h_neg=init_, h_pos= final_, n=2)
        plt.errorbar(1-imp, ROAR_metric, yerr= ROAR_err, c= kolors[0], label= 'ROAR', marker= 'h')
        plt.errorbar(1-imp, RANDOM_metric, yerr= RANDOM_err, c= kolors[1], label='Occluded at random', marker= 'p')
        plt.xlabel('Fraction of pixels')
        plt.ylabel('Metric: ['+metric+']')
        plt.grid(True)
        plt.legend()
        plt.show()
        return [(ROAR_metric, ROAR_err), (RANDOM_metric, RANDOM_err)]
    
    
    def KAR(self, X, Y, S, model, metric= 'AUROC'):
    
        """
        Make a plot with the KAR score.
        
        X --> Instances (nsamples X features)
        S --> Correspinding Saliency Maps
        model --> ANN trained model
        
        ### ### ###
        METRIC IMPLEMENTED : AUC
        otherwise
        MCC
        
        #### ROAR IMPLEMENTATION
        
        1. DIVIDE DATA IN 10 FOLDS and evalaute ROAR; comapre the results with 10 different random occusions
        
        """
        
        ### percetage of salient pixels
        N_sal_pxl = 17
        imp = np.linspace(0, 1, N_sal_pxl, endpoint=True)
        
        ###
        KAR_metric= []
        KAR_err= []
        #
        RANDOM_metric= []
        RANDOM_err= []
        
        
        ###number of Time-points
        Ntime = X.shape[1]
        
        ###
        Kroar = KFold(n_splits= 5, shuffle= True, random_state= 19)
        
        for fold in range(N_sal_pxl):
            
            internal_metric = []
            internal_metric_rnd = []
            
            #for index_keep, index_use in Kroar.split(X):

                ##select the data and the saliency maps
            Xuse = X.copy() #X[index_keep]
            Yuse = Y.copy() #Y[index_keep]
            Suse = S.copy() #S[index_keep]

            ### saliency threshold
            q0 = imp[fold]
            thr = np.quantile(Suse, q= q0)
                
                #if q0 == 0:
                #    thr = np.zeros(thr.shape)
                #elif q0 == 1:
                #    thr= np.ones(thr.shape)
                    
            ###take a value larger than "threshold-saliency", i.e., any value lower than threshold
            mask = (Suse>=thr).astype(int)
            Xsalient = Xuse*mask
                
                
                #####
                #print(Xsalient.shape, Xsalient_at_random.shape, Yuse.shape)
                
            ###USE AUC METRIC
            if metric == 'AUROC':

                ##salient 
                Msal = roc_auc_score(y_score=model.predict(Xsalient, verbose= None), y_true= Yuse)
                internal_metric.append(Msal)

                ##random 
                mrnd= []
                for ll in range(1):
                    ####
                    npr.seed(6)
                    rnd_tokens = npr.choice([0, 1], size=Xuse.shape[1], p= [imp[fold], 1-imp[fold]])
                    mask_rnd = npr.choice([0, 1], size= Xuse.shape, p= [imp[fold], 1-imp[fold]])
                    Xsalient_at_random = Xuse*mask_rnd
                    #####
                    Mrnd = roc_auc_score(y_score=model.predict(Xsalient_at_random, verbose= None), y_true= Yuse)
                    mrnd.append(Mrnd)
                    internal_metric_rnd.append(np.mean(Mrnd))
                
            ###USE AUC
            elif metric == 'AURPC':

                ##salient 
                Msal = roc_auc_score(y_score=model.predict(Xsalient, verbose= None), y_true= Yuse)
                internal_metric.append(Msal)

                ##random 
                mrnd= []
                for ll in range(1):
                    ####
                    npr.seed(6)
                    rnd_tokens = npr.choice([0, 1], size=Xuse.shape[1], p= [1-imp[fold], imp[fold]])
                    mask_rnd = npr.choice([0, 1], size= Xuse.shape, p= [1-imp[fold], imp[fold]])
                    Xsalient_at_random = Xuse*mask_rnd
                    #####
                    Mrnd = average_precision_score(y_score=model.predict(Xsalient_at_random, verbose= None), y_true= Yuse)
                    mrnd.append(Mrnd)
                internal_metric_rnd.append(np.mean(Mrnd))
                
                
            elif metric =='MCC':
                    
                ##salient 
                Msal = matthews_corrcoef(y_pred=(model.predict(Xsalient, verbose= None)>=.5).astype(int), 
                    y_true= Yuse)
                internal_metric.append(Msal)

                ##random 
                mrnd= []
                for ll in range(1):
                    ####
                    npr.seed(6)
                    rnd_tokens = npr.choice([0, 1], size=Xuse.shape[1], p= [1-imp[fold], imp[fold]])
                    mask_rnd = npr.choice([0, 1], size= Xuse.shape, p= [1-imp[fold], imp[fold]])
                    Xsalient_at_random = Xuse*mask_rnd
                    #####
                    Mrnd = matthews_corrcoef(y_pred=(model.predict(Xsalient_at_random, verbose= None)>=.5).astype(int), 
                                                 y_true= Yuse)
                    mrnd.append(Mrnd)
                internal_metric_rnd.append(np.mean(Mrnd))
                    
            #####Save Results
            KAR_metric.append(np.mean(internal_metric))
            #KAR_err.append(sem(internal_metric))
            KAR_err.append(0.01)
            ###
            RANDOM_metric.append(np.mean(internal_metric_rnd))
            RANDOM_err.append(sem(internal_metric_rnd))
            ######
        
        #############
        #############
        ### MAKE PLOT
        plt.figure(1)
        ###make ROAR first
        init_, final_ = 360*6/90, 360*29/90
        kolors = sns.diverging_palette(h_neg=init_, h_pos= final_, n=2)
        plt.errorbar(1-imp, KAR_metric, yerr= 0.01, c= kolors[0], label= 'KAR', marker= 'h')
        plt.errorbar(1-imp, RANDOM_metric, yerr= RANDOM_err, c= kolors[1], label='Included at random', marker= 'p')
        plt.xlabel('Fraction of pixels')
        plt.ylabel('Metric: '+metric)
        plt.grid(True)
        plt.legend()
        plt.show()
        return [(KAR_metric, KAR_err), (RANDOM_metric, RANDOM_err)]
    
    
    def Most_Salient_Pattern(self, X, Saliency_map, time_scale= 8/24):

        """
        Extract the most Salient Patterns
        
        X--> Time-Series Instances
        Saliency_map --> Salinecy map of X
        time_scale --> Time Scale of the Salient Patterns. A NUmber between 0 and 1. Fraction of X to be extracted.
        """

        ###get the window (Convolution Window)
        A = int(X.shape[1]*time_scale) #Window Amplitude
        W = np.ones(A)
        W = W/W.sum()

        ###get the Saliency...the best Saliency Location
        sal = np.apply_along_axis(np.convolve, axis= 1, arr= Saliency_map, v= W, mode= 'valid')
        best_location = np.argmax(sal, axis= 1)

        ##edge of the best locations
        Nplus = best_location+A-1


        ###get the most salient patterns
        list_of_most_salient_patterns = [X[kk, best_location[kk]:Nplus[kk], :] for kk in range(sal.shape[0])]
        most_salient_patterns = np.array(list_of_most_salient_patterns)

        return most_salient_patterns
    
    def min_distance_feature(self, data, kpatterns):
    
        """Evaluate the minimal distance feature given the K-patterns"""

        min_distance_feat = []
        for Kps in kpatterns:
            lenght = data.shape[1]-Kps.shape[0]+1
            Klenght = Kps.shape[0]
            box = []
                        
            for ii in range(data.shape[0]):
                box.append([np.mean(np.abs(data[ii, kk:kk+Klenght, :] - Kps)) for kk in range(lenght)])

            min_distance_feat.append(np.vstack(box).min(axis= -1))

        return np.array(min_distance_feat).T
    
    
    def aggregate_pca(self, loadings, explained_variance, variance_per_chunk= [1/2, 3/4, 7/8, 15/16]):
    
        """Aggregate PCA components


        loadings --> PCA Loadings
        explained_variance --> the variance explaiend per loading
        vartiance_per_chunk --> how to aggegate the variance explained

        """

        ####
        edge_min = np.array([np.where(np.cumsum(explained_variance)<=ratio)[0][-1] for ratio in variance_per_chunk])
        edge_max = np.array([np.where(np.cumsum(explained_variance)>=ratio)[0][0] for ratio in variance_per_chunk])
        ####
        new_loadings = np.vstack([loadings[edge_min[ii]:edge_max[ii], :].sum(axis= 0) for ii in range(edge_min.shape[0])])
        return new_loadings

    
    
    def TimeSeriesPCA(self, ts, 
                      n_comps= 10, 
                      reduction_factor= 1, 
                      variance_explained= False, 
                      aggregate= False):

        """
        Make TimeSeries PCA Decomposition

        ts --> Time-Sereis Instances
        ncomps--> number of components
        reduction_factor --> Use PPA to reduce memory load when fitting PCA; inverse of number of segments
        variance explained --> print the expalained Variance
        """

        ### reduction factor utilizes the PPA to reduce the Time-Series lenght
        ### It's scope is to reduce the memort needed to encode the Time-Series instance
        ### Drawback: Some infomration might be lost during the procedure...
        if reduction_factor<1:

            lenght_segments = int(1/reduction_factor)
            ppa = PiecewiseAggregateApproximation(n_segments= int(ts.shape[1]*reduction_factor))
            ppa.fit(ts)
            Xred_ppa = ppa.transform(ts)

            ## make PCA
            pca= PCA(n_components=n_comps)
            XXX = Xred_ppa.reshape(Xred_ppa.shape[0], Xred_ppa.shape[1]*(Xred_ppa.shape[2]))
            pca.fit(XXX)
            explained_variance = pca.explained_variance_ratio_

            if variance_explained:
                print('Variance Explained:', pca.explained_variance_ratio_.sum().round(2))

            ###get loadings
            loadings= pca.components_
            feats = pca.transform(XXX)

            ### loadigns reshapes as Time-Series Instance
            if aggregate:
                loadings = self.aggregate_pca(loadings, pca.explained_variance_ratio_)
                feats = np.matmul(XXX, loadings.T)
            
            loadings = loadings.reshape((loadings.shape[0], Xred_ppa.shape[1], Xred_ppa.shape[2])) 
            
            ###trasform back to the original time-domain
            loadings = ppa.inverse_transform(loadings)
            
            #correct bug at the edge of PPA invertse transform
            loadings = np.pad(loadings[:, 0:loadings.shape[1]-lenght_segments, :], 
                              pad_width=((0, 0), (0, lenght_segments), (0, 0)), 
                              mode='edge')
            
            
    
            ### smooth irregularities!
            w_lenght = lenght_segments+1-lenght_segments%2
            loadings= np.apply_along_axis(savgol_filter, 
                                          axis= 1, 
                                          arr= loadings, 
                                          window_length= w_lenght, 
                                          polyorder= 3) 

            #### #### ####

        elif reduction_factor ==1:
            Xred_ppa = ts.copy()

            ## make PCA
            pca= PCA(n_components=n_comps)
            pca.fit(Xred_ppa.reshape(Xred_ppa.shape[0], Xred_ppa.shape[1]*Xred_ppa.shape[2]))
            explained_variance = pca.explained_variance_ratio_
            XXX = Xred_ppa.reshape(Xred_ppa.shape[0], Xred_ppa.shape[1]*Xred_ppa.shape[2])

            if variance_explained:
                print('Variance Explained:', pca.explained_variance_ratio_.sum().round(2))

            ###get loadings
            loadings= pca.components_
            feats = pca.transform(XXX)
            
            if aggregate:
                loadings = self.aggregate_pca(loadings, explained_variance)
                feats = np.matmul(XXX, loadings.T)

            ### loadigns reshapes as Time-Series Instance
            loadings = loadings.reshape((loadings.shape[0], ts.shape[1], ts.shape[2])) 

            ### smooth irregularities!
            #w_lenght = lenght_segments+1-lenght_segments%2
            #loadings= np.apply_along_axis(savgol_filter, 
            #                              axis= 1, 
            #                              arr= loadings, 
            #                              window_length= w_lenght, 
            #                              polyorder= 3) 

        else:
            raise ValueError('reduction_factor must be less or equal 1!')
            reduction_factor
        
        
        return loadings, feats.reshape((ts.shape[0], -1)), explained_variance
    
    
    def Fourier_energy_chunk(self, X, kind= 'middle'):
        
        """
        EXTRACT ENERGY OF A TIME-SERIES FROM A PRECOMPILED SPECTRAL BAND!
        """
        
        Y = X-X.mean()
        erg = np.abs(npf.fft(Y)[0:int(Y.size/2)+Y.size%2])**2
        ####
        if erg.sum() !=0:
            erg = erg/erg.sum()
        else:
            erg = np.zeros(erg.size)

        freqs = 2*npf.fftfreq(Y.size)[0:int(Y.size/2)+Y.size%2]
        ###

        if kind=='very_high':
            ERG = erg[freqs>4/5]
            return ERG.sum()
        elif kind=='high':
            ERG = erg[(freqs<=4/5)&(freqs>3/5)]
            return ERG.sum()
        elif kind=='middle':
            ERG = erg[(freqs<=3/5)&(freqs>2/5)]
            return ERG.sum()
        elif kind=='low':
            ERG = erg[(freqs<=2/5)&(freqs>1/5)]
            return ERG.sum()
        elif kind=='very_low':
            ERG = erg[freqs<=1/5]
            return ERG.sum()
        
    def eval_presence(self, tsi, weight):

        """Evaluate the relevance of a waveform (weight) along the Time-Series instance (tsi)"""
        
        eval_ = []

        for kk in range(tsi.shape[0]):

            XX = tsi[kk].copy()
            YY = weight/np.sqrt((weight**2).sum())
            conv = convolve2d(XX, YY, mode='valid').ravel()

            ###normalization factor based on the norm of the TSI chunks
            f_factor = np.array([np.sqrt((XX[ii:ii+YY.shape[0]]**2).sum()) for ii in range(conv.shape[0])])

            ###
            eval_.append((conv/(f_factor)).max())

        return np.array(eval_)
    
    def eval_presence_1d(self, tsi, weight):

        """Evaluate the relevance of a waveform (weight) along the Time-Series instance (tsi)"""
        
        eval_ = []

        for kk in range(tsi.shape[0]):

            XX = tsi[kk].copy()
            YY = weight/np.sqrt((weight**2).sum())
            conv = convolve2d(XX, YY, mode='valid').ravel()

            ###normalization factor based on the norm of the TSI chunks
            f_factor = np.array([np.sqrt((XX[ii:ii+YY.shape[0]]**2).sum()) for ii in range(conv.shape[0])])

            ###
            eval_.append((conv/(f_factor)))

        return np.array(eval_)

    #def find_K_pattern(self, Z, ncomp):

    #    """Z --> Most salinet patterns 
        
    #       ncomp --> number of components
    #    """

    #    ###flattern
    #    Zshape = Z.shape
    #    Z = Z.reshape(Z.shape[0], -1)

        ####
        #ncomp = self.Optimal_number_of_components(Z)

        ### fit the K-means
        #kmeans = KMeans(n_clusters= ncomp, random_state= 8)
    #    kmeans = GaussianMixture(n_components=ncomp, random_state= 8)
    #    kmeans.fit(Z)

        ###get the K-patterns
        #kpattern = kmeans.cluster_centers_
    #    kpattern = kmeans.means_
        
        ###return the K-patterns with their original tensor shape 
    #    return kpattern.reshape(-1, Zshape[1], Zshape[2])
    
    def find_K_patterns(self, Z, ncomp):
        
        kshape= KShape(n_clusters= ncomp, random_state= 6, max_iter= 1000)
        kshape.fit(Z)
        
        Kpatterns = kshape.cluster_centers_
        Klables = kshape.labels_
        
        return Kpatterns, Klables
    
    def Kpattern_feature_extraction(self, X, kpattern):

        """Extract Space-Time Feature like in the shapelet transform but throught the K-patterns

           X --> the data
           kpattern --> kpattern

        """

        ### ### ###
        N = X.shape[1]
        M = kpattern[0].shape[0]
        nchannels = X.shape[2]


        space_feat = []
        time_feat = []
        for item in kpattern:

            k_list= [ np.sum(np.sum((X[:, kk:kk+M, :]-item)**2, axis= -1), axis=-1) for kk in range(N-M-1)]

            ### ### ###
            k_space = np.min(np.vstack(k_list), axis= 0)/(nchannels*M)
            k_time = np.argmin(np.vstack(k_list), axis= 0)/N

            space_feat.append(k_space)
            time_feat.append(k_time)

        ###create some names
        space_feat_name = ["presence_"+str(kk) for kk in range(len(space_feat))]
        time_feat_name = ["time_"+str(kk) for kk in range(len(time_feat))]

        #### #### ####
        extracted_features = np.hstack([np.vstack(space_feat).T, np.vstack(time_feat).T])
        extracted_features_names = np.hstack([space_feat_name, time_feat_name])

        return extracted_features, extracted_features_names
    
    
    def Optimal_number_of_components(self, X, Y, Z, 
                           use_smoteenn= False,
                           plot_results= False,
                           print_results= False,
                           save_plot= False):
    
        """The optimal number of components is that number of components such that the Logistic regression problem is solved at best.... 
        
        X --> the data
        Y --> their labels
        Z --> their most salinet patterns (of X)
        
        """

        acc = []
        acc_ci= [] 
        ncomp_ = np.arange(2, 16, 1).astype(int)
        
        ########################################### 
        ## DO NOT RUN ############################# 
        ########################################### 
        for ncomp in ncomp_:

            ####for a selected number of components 
            #1. get the K_patterns (a tensor with patterns)
            kpatterns, klables = self.find_K_patterns(Z, ncomp)
            
            #2. get the K-features
            new_x = self.min_distance_feature(X, kpatterns)
            new_y = klables.copy()
            
            if use_smoteenn:
                smotenn = SMOTEENN(sampling_strategy="auto")
                new_x, new_y = smotenn.fit_resample(new_x, new_y)
            
            
            #3. Fit a LR model
            #lr = LogisticRegressionCV(Cs= [1], scoring='brier_score_loss', 
            #                          cv= 5, 
            #                          fit_intercept=True)
            #lr.fit(new_x, new_y)
            silh = silhouette_samples(new_x, new_y)
            
            jack_result = jackknife_stats(silh, statistic= np.mean)
            acc.append(jack_result[0])
            acc_ci.append(jack_result[2])
          
            
        if print_results:
            print('Optimal number of components:', ncomp_[np.argmax(np.array(silh))])

        if plot_results:
            plt.errorbar(ncomp_, 
                         np.array(acc), 
                         yerr=np.array(acc_ci), 
                         color= 'tab:brown', marker= 'H')
            plt.ylabel('Silhouette Score')
            plt.xlabel('Number of K_components')
            plt.grid(True)
            if save_plot:
                plt.savefig("Optimal_number_of_Kcomponents.pdf")
            plt.show()

        return ncomp_[np.argmax(np.array(silh))]
    
    
class ModelSelection:
    
    def __init__(self):
        
        return
    
    
    def LSTM(self, X0, lstm_units= 8, dense_units= 1, lr= 1e-3):
        
        ####make a preprocessing with an LSTM; then use a Dense Layer to output the score
        
        ### Input
        Inputs = tf.keras.Input(shape=(X0.shape[1], X0.shape[2], ))
        lstm_ = tf.keras.layers.LSTM(units= lstm_units)(Inputs)
        Output = tf.keras.layers.Dense(units= dense_units, 
                                       activation= 'sigmoid')(lstm_)
        
        ### Define the model
        Model = tf.keras.models.Model(Inputs, Output)
        
        
        ###
        adam = tf.keras.optimizers.Adam(learning_rate= lr)
        bce = tf.keras.losses.BinaryCrossentropy()
        sce = tf.keras.losses.SparseCategoricalCrossentropy()
        Model.compile(optimizer= adam,
                        loss = bce)
        return Model        
        
    def GRU(self, X0, gru_units= 8, dense_units= 1, lr= 1e-3):
        
        ####make a preprocessing with an LSTM; then use a Dense Layer to output the score
        
        ### Input
        Inputs = tf.keras.Input(shape=(X0.shape[1], X0.shape[2]))
        gru_ = tf.keras.layers.GRU(units= gru_units)(Inputs)
        Output = tf.keras.layers.Dense(units= dense_units, activation= 'sigmoid')(gru_)
        
        ### Define the model
        Model = tf.keras.models.Model(Inputs, Output)
        
        ###
        adam = tf.keras.optimizers.Adam(learning_rate= lr)
        bce = tf.keras.losses.BinaryCrossentropy()
        sce = tf.keras.losses.SparseCategoricalCrossentropy()
        Model.compile(optimizer= adam,
                        loss = bce)
        return Model        
        
    
    def CNN_LSTM(self,
                 X0,
                 filters= 8, 
                 kernel_size= 5, 
                 lstm_units = 8,
                 poolsize= 2,
                 DropOut= 50e-2,
                 strides= 1, 
                 activation= 'relu', 
                 padding = 'valid', 
                 bias = True, 
                 activation_pred= 'sigmoid', 
                 deepness= 3, 
                 dense_units= 1,      
                 lr = 1e-3, 
                 print_summary= False):
        
        #######################################
        ###Define the CNN model
        ######################################
        
        ### Input
        Inputs = tf.keras.Input(shape=(X0.shape[1], X0.shape[2]))

        ### ### ### ### ###
        ### 1st_layer ## ##
        X = tf.keras.layers.Conv1D(filters= filters, 
                              kernel_size= kernel_size,
                              activation= 'linear', 
                              strides = strides, 
                              padding = padding, 
                              use_bias = bias, 
                              kernel_constraint= SemiOrthogonal())(Inputs)
        
        if activation == 'rational':
            X = RationalLayer()(X)
        else:
            X = tf.keras.layers.Activation(activation)(X)
        
        X = tf.keras.layers.MaxPool1D(pool_size = poolsize, padding= 'valid')(X)
        X = tf.keras.layers.Dropout(rate = DropOut)(X)

        ###other layers
        for jj in range(1, deepness):
            X = tf.keras.layers.Conv1D(filters= filters, 
                            kernel_size= kernel_size,
                            activation= 'linear', 
                            strides = strides, 
                            padding = padding, 
                            use_bias = bias, 
                            kernel_constraint= SemiOrthogonal())(X)
            
            if activation == 'rational':
                X = RationalLayer()(X)
            else:
                X = tf.keras.layers.Activation(activation)(X)
            #X = MUAF()(X)
            #X = RationalLayer()(X)
            X = tf.keras.layers.MaxPool1D(pool_size = poolsize, padding= 'valid')(X)
            X = tf.keras.layers.Dropout(rate = DropOut)(X)

        ### flattern   
        X = tf.keras.layers.LSTM(units= lstm_units)(X)

        ###make final prediction (with Platt's calibration)
        Xfinal = tf.keras.layers.Dense(units = 32, activation= activation_pred)(X) 
        Xfinal = RationalLayer()(Xfinal)
        Xfinal = tf.keras.layers.Dense(units = dense_units, activation= 'linear', use_bias= True)(Xfinal) 
        Xfinal = tf.keras.layers.Activation('sigmoid')(Xfinal)

        ##define model
        mymodel = tf.keras.models.Model(Inputs, Xfinal)
        #print(mymodel.summary())

        ###PRINT MODEL's SUMMARY
        if print_summary:
            print(mymodel.summary())

        ### Optimizer ADAM+SCE
        adam = tf.keras.optimizers.Adam(learning_rate= lr)
        bce = tf.keras.losses.BinaryCrossentropy()
        sce = tf.keras.losses.SparseCategoricalCrossentropy()
        mymodel.compile(optimizer= adam,
                        loss = bce)
        
        return mymodel
    
    
    def CNN_GRU(self,
                X0,
                 filters= 8, 
                 kernel_size= 5, 
                 gru_units = 8,
                 poolsize= 2,
                 DropOut= 50e-2,
                 strides= 1, 
                 activation= 'relu', 
                 padding = 'valid', 
                 bias = True, 
                 activation_pred= 'sigmoid', 
                 deepness= 3, 
                 dense_units= 1,      
                 lr = 1e-3, 
                 print_summary= False):
        
        #######################################
        ###Define the CNN model
        ######################################
        
        ### Input
        Inputs = tf.keras.Input(shape=(X0.shape[1], X0.shape[2]))

        ### ### ### ### ###
        ### 1st_layer ## ##
        X = tf.keras.layers.Conv1D(filters= filters, 
                              kernel_size= kernel_size,
                              activation= 'linear', 
                              strides = strides, 
                              padding = padding, 
                              use_bias = bias, 
                              kernel_constraint= SemiOrthogonal())(Inputs)
        
        if activation == 'rational':
            X = RationalLayer()(X)
        else:
            X = tf.keras.layers.Activation(activation)(X)
        
        X = tf.keras.layers.MaxPool1D(pool_size = poolsize, padding= 'valid')(X)
        X = tf.keras.layers.Dropout(rate = DropOut)(X)

        ###other layers
        for jj in range(1, deepness):
            X = tf.keras.layers.Conv1D(filters= filters, 
                            kernel_size= kernel_size,
                            activation= 'linear', 
                            strides = strides, 
                            padding = padding, 
                            use_bias = bias, 
                            kernel_constraint= SemiOrthogonal())(X)
            
            if activation == 'rational':
                X = RationalLayer()(X)
            else:
                X = tf.keras.layers.Activation(activation)(X)
            #X = MUAF()(X)
            #X = RationalLayer()(X)
            X = tf.keras.layers.MaxPool1D(pool_size = poolsize, padding= 'valid')(X)
            X = tf.keras.layers.Dropout(rate = DropOut)(X)

        ### flattern   
        X = tf.keras.layers.GRU(units= gru_units)(X)

        ###make final prediction (with Platt's calibration)
        Xfinal = tf.keras.layers.Dense(units = 32, activation= activation_pred)(X) 
        Xfinal = RationalLayer()(Xfinal)
        Xfinal = tf.keras.layers.Dense(units = dense_units, activation= 'linear', use_bias= True)(Xfinal) 
        Xfinal = tf.keras.layers.Activation('sigmoid')(Xfinal)

        ##define model
        mymodel = tf.keras.models.Model(Inputs, Xfinal)
        #print(mymodel.summary())

        ###PRINT MODEL's SUMMARY
        if print_summary:
            print(mymodel.summary())

        ### Optimizer ADAM+SCE
        adam = tf.keras.optimizers.Adam(learning_rate= lr)
        bce = tf.keras.losses.BinaryCrossentropy()
        sce = tf.keras.losses.SparseCategoricalCrossentropy()
        mymodel.compile(optimizer= adam,
                        loss = bce)
        
        return mymodel
        