#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import os
import csv
import math
import numpy
from matplotlib import pyplot
import joblib

def output(dirname, filename):
	print(filename)
	with open(dirname + "/" + filename, "r") as f:
		data = [[math.sqrt(float(line["x"])**2 + float(line["z"])**2), float(line["p"])] for line in csv.DictReader(f, skipinitialspace="True")]

	data = numpy.array(data).T
	pyplot.xlim([0, 0.06])
	pyplot.ylim([0, 600])
	pyplot.xlabel("r [m]")
	pyplot.ylabel("p [Pa]")
	pyplot.grid(which="minor", color="lightgray", linestyle="-")
	pyplot.grid(which="major", color="black", linestyle="-", b=True)
	pyplot.minorticks_on()
	pyplot.plot(data[0], data[1], ".")
	pyplot.savefig("output/" + filename + ".svg")
	pyplot.clf()

def main(dirname):
	joblib.Parallel(n_jobs=-1)([joblib.delayed(output)(dirname, filename) for filename in sorted(os.listdir(dirname))])

if __name__ == "__main__":
	main(sys.argv[1])
