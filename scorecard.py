from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy import sqrt, diag
from scipy.stats import norm
from copy import deepcopy

import pandas as pd
from pandas import DataFrame

class StepwiseLogisticRegression(BaseEstimator, ClassifierMixin):
	def __init__(self, sle = 0.01, sls = 0.05, verbose = 0, backward = True, forward = True, startFull = False, startList = None, remove_count = 1):
		self.sle = sle
		self.sls = sls
		self.verbose = verbose
		self.backward = backward
		self.forward = forward
		self.startFull = startFull
		self.startList = startList
		self.remove_count = remove_count
	def fit(self, X, y):
		self.mask = [1 for i in range(X.shape[1])]
		self.columns = [i for i in range(X.shape[1]) if self.mask[i] == 1]
		logit = LogisticRegression() #.fit(X[:, self.columns], y)
		# change the mask until convergence
		"""
		"""
		if self.startFull or not (self.forward or self.backward) and self.startList is None:
			# incomplete: make workable with arrays
			vnames = [colname for colname in X.columns]
		elif self.startList is not None:
			vnames = self.startList
		else:
			vnames = []
		bestnames = []
		bestname = ''
		iteration = 0;
		while True and (self.forward or self.backward):
			if self.verbose >= 1: print("Iteration %d" % iteration)
			changed = False
			
			if self.forward:
				p00 = 1.0
				for c in X.columns:
					if c not in vnames:
						vnames1 = deepcopy(vnames)
						vnames1.append(c)
						logit.fit(X[vnames1], y)
						p1 = logit_pval(logit, X[vnames1])
						if self.verbose >= 2: print("P-value of %s is % 2.4f" % (c, p1[-1]))
						if p1[-1] < p00:
							p00 = p1[-1]
							bestnames = vnames1
							bestname = c

				if p00 < self.sle:
					vnames = bestnames
					changed = True
					if self.verbose >= 1: print( 'Add ' + bestname)
				else:
					if self.verbose >= 1: print ("No variable to add at sle = %f" % self.sle)
			
			if self.backward:
				logit.fit(X[vnames], y)
				p0 = logit_pval(logit, X[vnames])[1:]
				for tmp in range(self.remove_count):
					if max(p0) > self.sls:
						idx = argmax(p0)
						if verbose >= 1: print ("Drop %s with p-value %2.4f" %(vnames[idx], max(p0)))
						vnames.pop(idx)
						p0 = delete(p0,idx)
						changed = True
			
			if not changed: 
				if self.verbose >= 1: print("No variables to add or remove, estimation complete")
				break
			logit.fit(X[vnames], y)
			logit.p = logit_pval(logit, X[vnames])
			if self.verbose >= 1:
				print('Current coefficients ' + str(vnames))
				print('Current p-values ' + str(p0))
				print("")
			iteration = iteration + 1
			
			
		logit.fit(X[vnames], y)
		logit.pval = logit_pval(logit, X[vnames])
		logit.variables = vnames
		
		self.logit = logit
		
		return self
    
def logit_pval(mdl, x):
	p = mdl.predict_proba(x)
	n = len(p)
	m = len(mdl.coef_[0]) + 1
	coefs = np.concatenate([mdl.intercept_, mdl.coef_[0]])
	Xmat= np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
	ans = np.zeros((m, m))
	for i in range(n):
		ans = ans + np.dot(np.transpose(Xmat[i, :]), Xmat[i, :]) * p[i,1] * p[i, 0]
	vcov = np.linalg.inv(np.matrix(ans))
	se = sqrt(diag(vcov))
	t =  coefs/se  
	p = (1 - norm.cdf(abs(t))) * 2
	return p
		

class Scorecard(BaseEstimator, ClassifierMixin):
	def __init__(self, selection = True, preliminary = True, transformation = 'logodds', cutoff = 0.5):
		self.selection = selection
		self.preliminary = preliminary
		self.transformation = transformation
		self.cutoff = cutoff
	def fit(self, X, y):
		X_clean = self.preprocess(X)
		self.k = X_clean.shape[1]
		self.fit_individual_scores(X_clean, y)
		X_tran = self.transform(X_clean)
		if self.selection not in ['forward', 'backward', 'stepwise', 'lasso']:
			self.logit = LogisticRegression.fit(X_tran, y)
		elif self.selection == 'lasso':
			self.logit = LogisticRegression.fit(X_tran, y)
		else:
			self.logit = StepwiseLogisticRegression(selection = self.selection).fit(X_tran, y)
		return self
	def preprocess_fit(self, X):
		# some creativity
		return preprocess(self, X)
	def preprocess(self, X):
		# replace zeroes
		# do something with categorical variables
		return X
	def fit_individual_scores(self, X, y):
		pass
	def transform(self, X):
		# if X is DataFrame, take its .values
		if self.preliminary == True:
			X_tran = pd.DataFrame([self.trees[i].predict_proba(X[:,i]) for i in range(self.k)]).transpose()
			if self.transformation == 'logodds':
				X_tran = np.log(X_tran/(1-X_tran))
		else:
			X_tran = pd.DataFrame([(X[:,i]>cut)*1 for i in range(self.k) for cut in self.cuts[i]]).transpose()
		# name columns in the frame
		return(X_tran)
	def predict_proba(self, X):
		if self.selection:
			proba = self.logit.predict_proba(self.transform(X))
		else:
			proba = self.logit.predict_proba(self.transform(X[:, self.selected_columns]))
		return proba
	def predict(self, X):
		proba1 = self.predict_proba[X][:,1]
		return (proba1 > self.cutoff) * 1
	def describe_scorecard(self):
		# return variables, bins and scores, if possible, in the form of a dataframe
		pass
