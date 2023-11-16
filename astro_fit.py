from astropy.modeling import models, fitting
from astropy.convolution import convolve_models

import numpy as np
import matplotlib.pyplot as plt

# create a single exGauss model
g = models.Gaussian1D(bounds={'amplitude': [0, 5], 'mean': [-300, 300], 'stddev': [0, 500]})
e = models.Exponential1D(amplitude=1, bounds={'tau': [0, 50]})
exgauss = convolve_models(g, e)

n = 3

# create the sum of n exGauss models
for _ in range(n):
	exgauss += exgauss

# print(exgauss.param_names)

for i in range(n-1):	
	# tie all tau parameters together
	getattr(exgauss, f'tau_{2*i+1}').tied = lambda model: model.tau_1

	# fix exponential amplitudes to 1
	getattr(exgauss, f'amplitude_{2*i+1}').fixed = True

# xs = np.arange(-100, 100, 0.01)
# ys = exgauss.evaluate(xs, 1, 0, 1, 1, -1, 1, 20, 1, 1, -1)

frb = 'data/single_FRB_221106_I_ts_343.0_64_avg_1_200.npy'

ys = np.load(frb)
xs = np.arange(len(ys)) + len(ys)/2

fitter = fitting.SLSQPLSQFitter()
fit = fitter(exgauss, xs, ys)

plt.plot(xs, ys)
plt.plot(xs, fit)
plt.show(block=True)