#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import os
import csv
import math
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import joblib

def main(dirname, dt, L):
	DT=0.1
	N = 10
	fileInterval = math.floor(DT/dt)
	files = sorted(os.listdir(dirname))
	for i in range(1, N+1):
		print(i)
		with open(dirname + "/" + files[i*fileInterval]) as f:
			actual = [[float(line["x"]), float(line["z"])] for line in csv.DictReader(f, skipinitialspace="True")]
		actual = numpy.array(actual).T
		pyplot.plot(actual[0], actual[1], ".")

		with open("koshizukaoka1996_shape/{0:02d}.csv".format(i)) as f:
			expected = [[float(line["x"]), float(line["z"])] for line in csv.DictReader(f, skipinitialspace="True")]
		expected = numpy.array(expected).T * 1e-2
		pyplot.plot(expected[0], expected[1], "-")

		pyplot.title("$t$={0:.1f}[s]".format(i*DT))
		pyplot.xlim([0, 4*L])
		pyplot.ylim([0, 2*L])
		pyplot.xlabel("$x$ [m]")
		pyplot.ylabel("$z$ [m]")
		pyplot.grid(which="minor", color="gray", linestyle="dashed")
		pyplot.grid(which="major", color="black", linestyle="solid", b=True)
		pyplot.minorticks_on()
		pyplot.savefig("koshizukaoka1996_shape_{0:02d}.svg".format(i))
		pyplot.clf()

if __name__ == "__main__":
	main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
