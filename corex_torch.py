import numpy as np  # Tested with 1.8.0
from torch import logsumexp
# from scipy.special import logsumexp  # Tested with 0.13.0
import torch

# Check if GPU is available 
try:
	dev_name = torch.cuda.get_device_name()
	print('pyTorch GPU available',dev_name)
except:
	print('Install Cuda and pyTorch for GPU computation.')


class Corex(object):
	"""
	Correlation Explanation
	A method to learn a hierarchy of successively more abstract
	representations of complex data that are maximally
	informative about the data. This method is unsupervised,
	requires no assumptions about the data-generating model,
	and scales linearly with the number of variables.
	Greg Ver Steeg and Aram Galstyan. "Discovering Structure in
	High-Dimensional Data Through Correlation Explanation."
	NIPS, 2014. arXiv preprint arXiv:1406.1222.
	Code follows sklearn naming/style (e.g. fit(X) to train)

	This version is an adaptation with Torch library to speed up the computation of CorEx.
	The code follow exactly the original code.

	Parameters
	----------
	n_hidden : int, optional, default=2
		Number of hidden units.
	dim_hidden : int, optional, default=2
		Each hidden unit can take dim_hidden discrete values.
	alpha_hyper : tuple, optional
		A tuple of three numbers representing hyper-parameters
		of the algorithm. See NIPS paper for meaning.
		Not extensively tested, but problem-specific tuning
		does not seem necessary.
	max_iter : int, optional
		Maximum number of iterations before ending.
	batch_size : int, optional
		Number of examples per minibatch. NOT IMPLEMENTED IN THIS VERSION.
	verbose : int, optional
		The verbosity level. The default, zero, means silent mode. 1 outputs TC(X;Y) as you go
		2 output alpha matrix and MIs as you go.
	seed : integer or numpy.RandomState, optional
		A random number generator instance to define the state of the
		random permutations generator. If an integer is given, it fixes the
		seed. Defaults to the global numpy random number generator.
	Attributes
	----------
	labels : array, [n_hidden, n_samples]
		Label for each hidden unit for each sample.
	clusters : array, [n_variables]
		Cluster label for each input variable.
	p_y_given_x : array, [n_hidden, n_samples, dim_hidden]
		The distribution of latent factors for each sample.
	alpha : array-like, shape (n_components,)
		Adjacency matrix between input variables and hidden units. In range [0,1].
	mis : array, [n_hidden, n_variables]
		Mutual information between each variable and hidden unit
	tcs : array, [n_hidden]
		TC(X_Gj;Y_j) for each hidden unit
	tc : float
		Convenience variable = Sum_j tcs[j]
	tc_history : array
		Shows value of TC over the course of learning. Hopefully, it is converging.
	References
	----------
	[1]     Greg Ver Steeg and Aram Galstyan. "Discovering Structure in
			High-Dimensional Data Through Correlation Explanation."
			NIPS, 2014. arXiv preprint arXiv:1406.1222.
	"""
	def __init__(self, n_hidden=2, dim_hidden=2,            # Size of representations
				 batch_size=1e6, max_iter=400, n_repeat=1,  # Computational limits
				 eps=1e-6, alpha_hyper=(0.3, 1., 500.), balance=0.,     # Parameters
				 missing_values=-1, seed=None, verbose=False,use_GPU = False, print_device = False):

		self.dim_hidden = dim_hidden  # Each hidden factor can take dim_hidden discrete values
		self.n_hidden = n_hidden  # Number of hidden factors to use (Y_1,...Y_m) in paper
		self.missing_values = missing_values  # Implies the value for this variable for this sample is unknown

		self.max_iter = max_iter  # Maximum number of updates to run, regardless of convergence

		self.eps = eps  # Change in TC to signal convergence
		self.lam, self.tmin, self.ttc = alpha_hyper  # Hyper-parameters for updating alpha
		self.balance = balance # 0 implies no balance constraint. Values between 0 and 1 are valid.

		# if type(seed) == int:
		np.random.seed(seed)# Set for deterministic results
		# torch.manual_seed(seed) # Set for deterministic results

		if torch.cuda.is_available() and use_GPU:
			self.device = torch.device("cuda")
			torch.cuda.set_device(0)
			if print_device:
				print("Device using : " + torch.cuda.get_device_name(0))
		else:
			if print_device:
				print("torch CPU uses")
			self.device = torch.device("cpu")

		self.verbose = verbose
		if verbose > 0:
			np.set_printoptions(precision=3, suppress=True, linewidth=200)
			print('corex, rep size: {}, {}'.format(n_hidden, dim_hidden))
		if verbose > 1:
			np.seterr(all='warn')
		else:
			np.seterr(all='ignore')

	def label(self, p_y_given_x):
		"""Maximum likelihood labels for some distribution over y's"""
		res = torch.argmax(p_y_given_x, dim=2).T
		if self.device == torch.device('cuda'):
			res = res.cpu()
		return res.numpy()

	@property
	def labels(self):
		"""Maximum likelihood labels for training data. Can access with self.labels (no parens needed)"""
		return self.label(self.p_y_given_x)

	@property
	def clusters(self):
		"""Return cluster labels for variables"""
		res = torch.argmax(self.alpha[:,:,0],dim=0)
		if self.device == torch.device('cuda'):
			res = res.cpu()
		return res.numpy()

	@property
	def tc(self):
		"""The total correlation explained by all the Y's.
		(Currently correct only for trees, modify for non-trees later.)"""
		res = torch.sum(self.tcs)
		if self.device == torch.device('cuda'):
			res = res.cpu()
		return res.item()

	def event_from_sample(self, x):
		"""Transform data into event format.
		For each variable, for each possible value of dim_visible it could take (an event),
		we return a boolean matrix of True/False if this event occurred in this sample, x.
		Parameters:
		x: {array-like}, shape = [n_visible]
		Returns:
		x_event: {array-like}, shape = [n_visible * self.dim_visible]
		"""

		x = np.asarray(x)
		n_visible = x.shape[0]

		assert self.n_visible == n_visible, \
			"Incorrect dimensionality for samples to transform."

		return np.ravel(x[:, np.newaxis] == np.tile(np.arange(self.dim_visible), (n_visible, 1)))

	def events_from_samples(self, X):
		"""Transform data into event format. See event_from_sample docstring."""
		n_samples, n_visible = X.shape
		events_to_transform = np.empty((self.n_events, n_samples), dtype=bool)
		for l, x in enumerate(X):
			events_to_transform[:, l] = self.event_from_sample(x)
		return events_to_transform

	def transform(self, X, details=False):
		"""
		Label hidden factors for (possibly previously unseen) samples of data.
		Parameters: samples of data, X, shape = [n_samples, n_visible]
		Returns: , shape = [n_samples, n_hidden]
		"""
		if X.ndim < 2:
			X = X[np.newaxis, :]
		events_to_transform = self.events_from_samples(X)
		events_to_transform = torch.tensor(events_to_transform,device=self.device)
		p_y_given_x, log_z = self.calculate_latent(events_to_transform)
		if details:
			return p_y_given_x, log_z
		else:
			return self.label(p_y_given_x)

	def transform_proba(self,X):
		p_y_given_x,_ = self.transform(X,details=True)
		return p_y_given_x


	def fit(self, X, **params):
		"""Fit CorEx on the data X.
		Parameters
		----------
		X: {array-like, sparse matrix}, shape = [n_samples, n_visible]
			Data matrix to be
		Returns
		-------
		self
		"""
		self.fit_transform(X)
		return self

	def fit_transform(self, X):
		"""Fit corex on the data (this used to be ucorex)
		Parameters
		----------
		X : array-like, shape = [n_samples, n_visible]
			The data.
		Returns
		-------
		Y: array-like, shape = [n_samples, n_hidden]
		   Learned values for each latent factor for each sample.
		   Y's are sorted so that Y_1 explains most correlation, etc.
		"""

		self.initialize_parameters(X)

		X_event = self.events_from_samples(X)  # Work with transformed representation of data for efficiency
		X_event = torch.tensor(X_event,device=self.device)

		self.X_event = X_event

		self.p_x, self.entropy_x = self.data_statistics(X_event)
		
		for nloop in range(self.max_iter):

			self.update_marginals(X_event, self.p_y_given_x)  # Eq. 8

			if self.n_hidden > 1:  # Structure learning step
				self.mis = self.calculate_mis(self.log_p_y, self.log_marg)
				self.update_alpha(self.mis, self.tcs)  # Eq. 9

			self.p_y_given_x, self.log_z = self.calculate_latent(X_event)  # Eq. 7

			self.update_tc(self.log_z)  # Calculate TC and record history for convergence

			self.print_verbose()
			if self.convergence(): break

		self.sort_and_output()

		del self.X_event # Free memory. Running CorEx several times could lead to full memory for GPU/CPU

		return self.labels

	def fit_increm(self,x_new,random=False):

		# Set a new X_event
		x = np.asarray(x_new)
		visible_values = np.unique(x)
		dim_visible_new = np.max(visible_values) + 1
		dim_visible_new = max(self.dim_visible,dim_visible_new)
		x_event = (x[:,np.newaxis] == np.tile(np.arange(dim_visible_new),(1,1))).T
		x_event = torch.tensor(x_event,device=self.device)
		X_event = torch.cat((self.X_event,x_event),0)
		self.X_event = X_event

		#Set new statistics
		self.p_x_prev,self.entropy_x_prev = self.p_x,self.entropy_x
		self.n_visible += 1
		self.p_x, self.entropy_x = self.data_statistics(X_event)

		#New alpha :
		# take the Argmax_{j} I(X_new,Y_j) and set to 1 and other to zeros.
		if not random:
			self.log_marg = self.calculate_p_y_xi(self.X_event, self.p_y_given_x)
			mis = self.calculate_mis(self.log_p_y,self.log_marg).view((self.n_hidden,self.n_visible))
			argmaxmis = torch.argmax(mis[:,-1])
			alpha_new = torch.zeros((self.n_hidden,1,1),device = self.device,dtype = torch.float)
			alpha_new[argmaxmis] = 1
		else:
			alpha_new = (0.5+0.5*np.random.random((self.n_hidden, 1, 1))) # random init
			alpha_new = torch.tensor(alpha_new,device = self.device,dtype = torch.float)	
		self.alpha = torch.cat((self.alpha,alpha_new),1)

		for nloop in range(self.max_iter):

			self.update_marginals(X_event, self.p_y_given_x)  # Eq. 8

			if self.n_hidden > 1:  # Structure learning step
				self.mis = self.calculate_mis(self.log_p_y, self.log_marg)
				self.update_alpha(self.mis, self.tcs)  # Eq. 9

			self.p_y_given_x, self.log_z = self.calculate_latent(X_event)  # Eq. 7

			self.update_tc(self.log_z)  # Calculate TC and record history for convergence

			self.print_verbose()
			if self.convergence(): break

		self.sort_and_output()

		return self

	def initialize_parameters(self, X):
		"""Set up starting state
		Parameters
		----------
		X : array-like, shape = [n_samples, n_visible]
			The data.
		"""

		self.n_samples, self.n_visible = X.shape
		self.initialize_events(X)
		self.initialize_representation()

	def initialize_events(self, X):
		values_in_data = set(np.unique(X).tolist())-set([self.missing_values])
		self.dim_visible = int(max(values_in_data)) + 1
		if not set(range(self.dim_visible)) == values_in_data:
			print("Warning: Data matrix values should be consecutive integers starting with 0,1,...")
		self.n_events = self.n_visible * self.dim_visible

	def initialize_representation(self):
		if self.n_hidden > 1:
			self.alpha = (0.5+0.5*np.random.random((self.n_hidden, self.n_visible, 1)))
			self.alpha = torch.tensor(self.alpha,device = self.device)
		else:
			self.alpha = torch.ones((self.n_hidden, self.n_visible, 1), dtype=torch.float,device = self.device)
		self.tc_history = []
		self.tcs = torch.zeros(self.n_hidden,device = self.device,dtype = torch.float)

		log_p_y_given_x_unnorm = -np.log(self.dim_hidden) * (0.5 + np.random.random((self.n_hidden, self.n_samples, self.dim_hidden)))
		log_p_y_given_x_unnorm = torch.tensor(log_p_y_given_x_unnorm,device = self.device)
		#log_p_y_given_x_unnorm = -100.*np.random.randint(0,2,(self.n_hidden, self.n_samples, self.dim_hidden))
		self.p_y_given_x, self.log_z = self.normalize_latent(log_p_y_given_x_unnorm)

	def data_statistics(self, X_event):
		p_x = torch.sum(X_event, dim=1,dtype = torch.float)
		p_x = p_x.view((self.n_visible, self.dim_visible))
		p_x /= torch.sum(p_x, dim=1, keepdim=True)  # With missing values, each x_i may not appear n_samples times
		z = torch.zeros(1,device = self.device)
		ep = torch.tensor(1e-10,device = self.device)
		entropy_x = torch.sum(torch.where(p_x>0., -p_x * torch.log(p_x), z), dim=1)
		entropy_x = torch.where(entropy_x > 0, entropy_x, ep)
		return p_x, entropy_x

	def update_marginals(self, X_event, p_y_given_x):
		self.log_p_y = self.calculate_p_y(p_y_given_x)
		self.log_marg = self.calculate_p_y_xi(X_event, p_y_given_x) - self.log_p_y

	def calculate_p_y(self, p_y_given_x):
		"""Estimate log p(y_j) using a tiny bit of Laplace smoothing to avoid infinities."""
		pseudo_counts = 0.001 + torch.sum(p_y_given_x.float(), dim=1, keepdim=True)
		log_p_y = torch.log(pseudo_counts) - torch.log(torch.sum(pseudo_counts, dim=2, keepdim=True))
		return log_p_y

	def calculate_p_y_xi(self, X_event, p_y_given_x):
		"""Estimate log p(y_j|x_i) using a tiny bit of Laplace smoothing to avoid infinities."""
		pseudo_counts = 0.001 + torch.matmul(X_event.float(), p_y_given_x.float())  # n_hidden, n_events, dim_hidden
		log_marg = torch.log(pseudo_counts) - torch.log(torch.sum(pseudo_counts, dim=2, keepdim=True))
		return log_marg  # May be better to calc log p(x_i|y_j)/p(x_i), as we do in Marg_Corex

	def calculate_mis(self, log_p_y, log_marg):
		"""Return normalized mutual information"""
		vec = torch.exp(log_marg + log_p_y)  # p(y_j|x_i)
		smis = torch.sum(vec * log_marg, dim=2)
		smis = smis.view((self.n_hidden, self.n_visible, self.dim_visible))
		mis = torch.sum(smis * self.p_x, dim=2, keepdim=True)
		return mis/self.entropy_x.view((1, -1, 1))

	def update_alpha(self, mis, tcs):
		t = (self.tmin + self.ttc * torch.abs(tcs)).view((self.n_hidden, 1, 1))
		maxmis = torch.max(mis, dim=0).values
		alphaopt = torch.exp(t * (mis - maxmis))
		self.alpha = (1. - self.lam) * self.alpha.float() + self.lam * alphaopt

	def calculate_latent(self, X_event):
		""""Calculate the probability distribution for hidden factors for each sample."""
		alpha_rep = self.alpha.repeat_interleave(repeats = self.dim_visible, dim=1)
		log_p_y_given_x_unnorm = (1. - self.balance) * self.log_p_y + torch.matmul(X_event.T.float(), alpha_rep*self.log_marg)

		return self.normalize_latent(log_p_y_given_x_unnorm)

	def normalize_latent(self, log_p_y_given_x_unnorm,is_np = False):
		"""Normalize the latent variable distribution
		For each sample in the training set, we estimate a probability distribution
		over y_j, each hidden factor. Here we normalize it. (Eq. 7 in paper.)
		This normalization factor is quite useful as described in upcoming work.
		Parameters
		----------
		Unnormalized distribution of hidden factors for each training sample.
		Returns
		-------
		p_y_given_x : 3D array, shape (n_hidden, n_samples, dim_hidden)
			p(y_j|x^l), the probability distribution over all hidden factors,
			for data samples l = 1...n_samples
		log_z : 2D array, shape (n_hidden, n_samples)
			Point-wise estimate of total correlation explained by each Y_j for each sample,
			used to estimate overall total correlation.
		"""
		if not is_np:
			log_z = torch.logsumexp(log_p_y_given_x_unnorm, dim=2)  # Essential to maintain precision.
			log_z = log_z.view((self.n_hidden, -1, 1))
			res = torch.exp(log_p_y_given_x_unnorm - log_z), log_z
		else:
			log_z = logsumexp(log_p_y_given_x_unnorm, axis=2)  # Essential to maintain precision.
			log_z = log_z.reshape((self.n_hidden, -1, 1))
			res = np.exp(log_p_y_given_x_unnorm - log_z), log_z
		return res

	def update_tc(self, log_z):
		self.tcs = torch.mean(log_z, dim=1).view(-1)
		sum_tcs = torch.sum(self.tcs)
		if self.device == torch.device('cuda'):
			sum_tcs = sum_tcs.cpu()
		self.tc_history.append(sum_tcs.item())

	def sort_and_output(self):
		order = torch.argsort(self.tcs,descending=True)  # Order components from strongest TC to weakest
		self.tcs = self.tcs[order]  # TC for each component
		self.alpha = self.alpha[order]  # Connections between X_i and Y_j
		self.p_y_given_x = self.p_y_given_x[order]  # Probabilistic labels for each sample
		self.log_marg = self.log_marg[order]  # Parameters defining the representation
		self.log_p_y = self.log_p_y[order]  # Parameters defining the representation
		self.log_z = self.log_z[order]  # -log_z can be interpreted as "surprise" for each sample
		if hasattr(self, 'mis'):
			self.mis = self.mis[order]

	def print_verbose(self):
		if self.verbose:
			print(self.tcs)
		if self.verbose > 1:
			print(self.alpha[:,:,0])
			if hasattr(self, "mis"):
				print(self.mis[:,:,0])

	def convergence(self):
		if len(self.tc_history) < 10:
			return False
		dist = -np.mean(self.tc_history[-10:-5]) + np.mean(self.tc_history[-5:])
		return np.abs(dist) < self.eps # Check for convergence. dist is nan for empty arrays, but that's OK

	def new_increm(self,x):
		if x.ndim > 1:
			x = x.reshape(-1)
		x_event = self.event_from_sample(x)
		n_new_samples = x_event.shape[0]
		x_event = torch.tensor(x_event,device=self.device)

		if n_new_samples != self.n_samples:
			print('Bootstrap needed')
			return self

		# representations init
		log_p_y_given_x_unnorm = -np.log(self.dim_hidden) * (0.5 + np.random.random((self.n_hidden, self.n_new_samples, self.dim_hidden)))
		log_p_y_given_x_unnorm = torch.tensor(log_p_y_given_x_unnorm,device = self.device)

		log_p_y_given_x_prev_unnorm = torch.log(torch.tensor(self.p_y_given_x,device  =self.device))

		log_p_y_given_x_unnorm = log_p_y_given_x_unnorm + log_p_y_given_x_prev_unnorm

		self.p_y_given_x, self.log_z = self.normalize_latent(log_p_y_given_x_unnorm)

	def calculate_logz(self,x):
		# p_y_given_x = self.transform_proba(X)
		# # event = torch.tensor(self.events_from_samples(x)).to(device=self.device)
		# alpha_rep = self.alpha.repeat_interleave(repeats = self.dim_visible, dim=1)
		# log_ratio = self.alpha*torch.log(p_y_given_x) - self.log_p_y

		# log_z = self.log_p_y + log_ratio 
		x_events = self.transform_proba(x)
		proba,log_z = self.calculate_latent(x_events)
		return log_z

		