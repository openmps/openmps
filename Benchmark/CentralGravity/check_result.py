#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import csv
import math
import numpy
from matplotlib import pyplot

def main(filename):
	with open(filename, "r") as f:
		data = [[math.sqrt(float(line["x"])**2 + float(line["z"])**2), float(line["p"])] for line in csv.DictReader(f, skipinitialspace="True")]

	data = numpy.array(data).T
	pyplot.plot(data[0], data[1], ".")
	pyplot.savefig("out.svg")
	pyplot.clf()

if __name__ == "__main__":
	main(sys.argv[1])
