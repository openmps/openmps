#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import csv
import math
from matplotlib import pyplot

def main(filename):
	r = []
	p = []
	with open(filename) as f:
		header = True
		for line in csv.reader(f, delimiter=","):
			if header:
				header = False
			else:
				x = float(line[1])
				z = float(line[2])
				r.append(math.sqrt(x*x + z*z))
				p.append(float(line[5]))
	
	pyplot.plot(r, p, ".")
	pyplot.savefig("out.svg")
	pyplot.clf()

if __name__ == "__main__":
	main(sys.argv[1])
