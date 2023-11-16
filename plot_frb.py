import numpy as np
import matplotlib.pyplot as plt

frb = np.load('data/single_FRB_221106_I_ts_343.0_64_avg_1_200.npy')[3700:4300]

plt.plot(range(len(frb)), frb, color='red')
plt.ylabel('Intensity')
plt.xlabel('Time (us)')
plt.title('FRB 221106')

plt.show(block=True)
