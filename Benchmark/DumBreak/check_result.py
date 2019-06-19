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

def calculate_n0(r_e):
	size = int(math.ceil(r_e))
	n0 = 0.0
	for y in range(-size, size+1):
		for x in range(-size, size+1):
			r = math.sqrt(x*x + y*y)
			if((0 < r) and (r < r_e)):
				w = r_e/r - 1
				n0 += w
	return n0

def output(dirname, filename):
	with open(dirname + "/" + filename, "r") as f:
		data = [[float(line["x"]), int(line["Type"])] for line in csv.DictReader(f, skipinitialspace="True")]

	edge = max([d[0] for d in data if d[1] == 0])

	return [edge]

def main(dirname, L, dt, g):
	data = joblib.Parallel(n_jobs=-1, verbose=1)([joblib.delayed(output)(dirname, filename) for filename in sorted(os.listdir(dirname))])
#	data = [output(dirname, filename) for filename in sorted(os.listdir(dirname)) if filename.startswith("particles_")]

	data = numpy.array(data).T
	n = data.shape[1]
	maxT = n * dt
	t = numpy.array(range(n)) * dt

	tt = math.sqrt(2*g/L)
	z = data[0]
	pyplot.plot(t*tt, z/L)
	pyplot.xlim([0, 3.5])
	pyplot.ylim([1, 4])
	pyplot.xlabel("$t\sqrt{2g/L}$")
	pyplot.ylabel("$Z/L$")
	pyplot.grid(which="minor", color="gray", linestyle="dashed")
	pyplot.grid(which="major", color="black", linestyle="solid", b=True)
	pyplot.minorticks_on()
	pyplot.savefig("edge.svg")
	pyplot.clf()

if __name__ == "__main__":
	main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
