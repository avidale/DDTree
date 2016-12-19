from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np



class BinningClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, max_bins = 5, initial_quantiles = 100, prior_obs = 100, loss = 'woe', link = 'linear', epsilon = 1):
		self.max_bins = max_bins
		self.initial_quantiles = initial_quantiles
		self.prior_obs = prior_obs
		self.loss = loss
		self.link = link
		self.epsilon = epsilon
	def fit(self, X, y):
		#if np.unique(y) != np.array([0,1]):
		self.categoricals = None
		self.column_names = None
		self.global_mean = np.mean(y)
		if type(X) == pd.DataFrame:
			''' Replace categorical values with ranks of their smoothed corresponding bad rates '''
			self.categoricals = dict()
			self.column_names = X.columns
			for i, c in enumerate(X.columns):
				if X[c].dtype == 'object':
					df = pd.DataFrame({'x':X[c], 'y':y})
					gr = df.groupby('x')['y'].aggregate([sum, len])
					gr['t'] = (gr['sum']+self.prior_obs*self.global_mean) / (gr['len']+self.prior_obs)
					self.categoricals[c] = gr['t'].rank().to_dict()
			X = X.replace(self.categoricals)
		X = np.array(X)
		self.n_obs, self.n_features = X.shape
		self.bins = []
		for i in range(self.n_features):
			''' transform feature to get some preliminary groups, like unique quantiles '''
			x = X[:,i]
			finebins = np.unique(x)
			if len(finebins) + 1 > self.initial_quantiles:
				finebins = np.unique(np.percentile(x, np.arange(0, 100, 100.0/(self.initial_quantiles+1))[:-1], interpolation = 'nearest'))
			# INCOMPLETE: need to react if only one unique value is available
			finebins = finebins[1:]
			x = np.digitize(x, finebins)
			coarsebins = finebins.copy()
			
			df = pd.DataFrame({'x':x, 'y':y}).groupby('x')['y'].aggregate([sum, len])
			while(len(coarsebins)+1) > self.max_bins:
				metric_max = -np.inf
				j_max = -1
				for j in range(len(coarsebins)):
					df2 = self._unify(df, j)
					m = self._evaluate_metric(df2, self.loss)
					if m > metric_max:
						metric_max = m
						j_max = j
				coarsebins = np.delete(coarsebins, j_max)
				df = self._unify(df, j_max)
			self.bins.append(coarsebins)
	def _unify(self, df, last_of_2):
		res = df.copy()
		res.iloc[last_of_2 - 1] = res.iloc[last_of_2 - 1] + res.iloc[last_of_2]
		return res.drop(res.index[last_of_2], axis = 0)
	def _evaluate_metric(self, df, metric):
		n = df['len'] + 2 * self.epsilon
		b = df['sum'] + self.epsilon
		g = n - b
		G = np.sum(g)
		B = np.sum(b)
		if metric == 'woe':
			woe = np.log(g/G/(b/B))
			res = np.sum((g/G-b/B)*woe)
		if metric == 'chi':
			res = None
		return res
			
	
	def transform(self, X):
		pass
	def predict(self, X):
		pass
	def predict_proba(self, X):
		pass




