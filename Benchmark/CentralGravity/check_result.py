#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import os
import csv
import math
import numpy
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

def output(dirname, filename, r_e, beta):
	print(filename)
	with open(dirname + "/" + filename, "r") as f:
		data = [[math.sqrt(float(line["x"])**2 + float(line["z"])**2), float(line["p"]), float(line["n"])] for line in csv.DictReader(f, skipinitialspace="True")]

	data = numpy.array(data)

	# pyplot.xlim([0, 0.06])
	# pyplot.ylim([0, 600])
	# pyplot.xlabel("r [m]")
	# pyplot.ylabel("p [Pa]")
	# pyplot.grid(which="minor", color="lightgray", linestyle="-")
	# pyplot.grid(which="major", color="black", linestyle="-", b=True)
	# pyplot.minorticks_on()
	# pyplot.plot(data.T[0], data.T[1], ".")
	# pyplot.savefig("output/" + filename + ".svg")
	# pyplot.clf()

	n0 = calculate_n0(r_e)
	n_surface = n0 * beta
	surface = [d[0] for d in data if d[2] < n_surface]
	r_max = max(surface)
	r_min = min(surface)
	return r_min/r_max

def main(dirname, r_e, beta):
	data = joblib.Parallel(n_jobs=-1)([joblib.delayed(output)(dirname, filename, r_e, beta) for filename in sorted(os.listdir(dirname))])
#	data = [output(dirname, filename, r_e, beta) for filename in sorted(os.listdir(dirname))]

	pyplot.plot(data)
	pyplot.savefig("roundness.svg")
	pyplot.clf()

if __name__ == "__main__":
	main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
