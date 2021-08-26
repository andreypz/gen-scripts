#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from coffea import hist

a = hist.Hist("Counts", hist.Bin("aaa", "", 50, -3, 3))
nn = np.random.normal(0,0.5,size=200)
ww = nn
a.fill(aaa=nn, weight=ww)
plt.gcf().clf()
hist.plot1d(a)
#hist.plot1d(a, line_opts={} )
#plt.ylim(-12, 12)
#plt.gca().relim()
plt.gca().autoscale()

plt.gcf().savefig("a.png")

import hist as default_hist

b = default_hist.Hist.new.Reg(50, -3, 3).Weight()

b.fill(nn, weight=ww)
plt.gcf().clf()
b.plot()
#plt.ylim(-12, 12)
plt.gcf().savefig("b.png")



