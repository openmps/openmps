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
import statistics

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

def output(dirname, filename, l_0, min_n):
	with open(dirname + "/" + filename, "r") as f:
		data = [[float(line["x"]), float(line["z"]), float(line["p"]), float(line["n"]), int(line["Type"])] for line in csv.DictReader(f, skipinitialspace="True")]

	X_H1 = 2020e-3 - 1525e-3
	X_H2 = 2020e-3 - 1028e-3
	Z_P2 = 160e-3
	D = 90e-3

	h1 = max([0] + [d[1] for d in data if abs(d[0] - X_H1) < l_0/2 and d[3] > min_n])
	h2 = max([0] + [d[1] for d in data if abs(d[0] - X_H2) < l_0/2 and d[3] > min_n])
	p2 = statistics.mean(d[2] for d in data if d[0] < 0 and d[4] == 1 and abs(d[1] - Z_P2) < D/2)

	return [h1, h2, p2]

def main(dirname, l_0, dt, r_e):
	H = 0.6
	g = 9.8
	rho = 1000
	tt = math.sqrt(g/H)
	p0 = rho*g*H

	n0 = calculate_n0(r_e)
	min_n = n0*0.1
	data = joblib.Parallel(n_jobs=-1, verbose=1)([joblib.delayed(output)(dirname, filename, l_0, min_n) for filename in sorted(os.listdir(dirname))])
	# data = [output(dirname, filename, l_0, min_n) for filename in sorted(os.listdir(dirname)) if filename.startswith("particles_00000")]

	data = numpy.array(data).T
	n = data.shape[1]
	maxT = n * dt
	t = numpy.array(range(n)) * dt

	h1 = data[0]
	pyplot.plot(t*tt, h1/H, '-', label="OpenMPS", linewidth=2)
	with open("zhouetal1999/h1.csv", "r") as f:
		exp = [[float(line["t"]), float(line["h"])] for line in csv.DictReader(f, skipinitialspace="True")]
	exp = numpy.array(exp).T
	pyplot.plot((exp[0] - exp[0][0])*tt, exp[1]/H, '--', label="Exp: Zhou et al. (1999)")
	with open("zhouetal1999/abdolmalekietal2004_h1.csv", "r") as f:
		fluent = [[float(line["t"]), float(line["h"])] for line in csv.DictReader(f, skipinitialspace="True")]
	fluent = numpy.array(fluent).T
	pyplot.plot(fluent[0], fluent[1], '-', label="Fluent: Abdolmaleki et al. (2004)")
	pyplot.xlim([0, 16])
	pyplot.ylim([0, 1.0])
	pyplot.xlabel(r"$t\sqrt{g/H}$")
	pyplot.ylabel(r"$h_1/H$")
	pyplot.grid(which="minor", color="gray", linestyle="dashed")
	pyplot.grid(which="major", color="black", linestyle="solid", b=True)
	pyplot.minorticks_on()
	pyplot.legend()
	pyplot.savefig("zhouetal1999_h1.svg")
	pyplot.clf()

	h2 = data[1]
	pyplot.plot(t*tt, h2/H, '-', label="OpenMPS", linewidth=2)
	with open("zhouetal1999/h2.csv", "r") as f:
		exp = [[float(line["t"]), float(line["h"])] for line in csv.DictReader(f, skipinitialspace="True")]
	exp = numpy.array(exp).T
	pyplot.plot((exp[0] - exp[0][0])*tt, exp[1]/H, '--', label="Exp: Zhou et al. (1999)")
	with open("zhouetal1999/abdolmalekietal2004_h2.csv", "r") as f:
		fluent = [[float(line["t"]), float(line["h"])] for line in csv.DictReader(f, skipinitialspace="True")]
	fluent = numpy.array(fluent).T
	pyplot.plot(fluent[0], fluent[1], '-', label="Fluent: Abdolmaleki et al. (2004)")
	pyplot.xlim([0, 16])
	pyplot.ylim([0, 1.0])
	pyplot.xlabel(r"$t\sqrt{g/H}$")
	pyplot.ylabel(r"$h_2/H$")
	pyplot.grid(which="minor", color="gray", linestyle="dashed")
	pyplot.grid(which="major", color="black", linestyle="solid", b=True)
	pyplot.minorticks_on()
	pyplot.legend()
	pyplot.savefig("zhouetal1999_h2.svg")
	pyplot.clf()

	p2 = data[2]
	pyplot.plot(t*tt, p2/p0, '-', label="OpenMPS")
	with open("zhouetal1999/p2.csv", "r") as f:
		exp = [[float(line["t"]), float(line["p"])] for line in csv.DictReader(f, skipinitialspace="True")]
	exp = numpy.array(exp).T
	pyplot.plot((exp[0] - exp[0][0])*tt, exp[1]/p0, '--', label="Exp: Zhou et al. (1999)")
	with open("zhouetal1999/abdolmalekietal2004_p2.csv", "r") as f:
		fluent = [[float(line["t"]), float(line["p"])] for line in csv.DictReader(f, skipinitialspace="True")]
	fluent = numpy.array(fluent).T
	pyplot.plot(fluent[0], fluent[1], '-', label="Fluent: Abdolmaleki et al. (2004) Vertex Average")
	pyplot.xlim([0, 16])
	pyplot.ylim([0, 1.2])
	pyplot.xlabel(r"$t\sqrt{g/H}$")
	pyplot.ylabel(r"$P/\rho g H$")
	pyplot.grid(which="minor", color="gray", linestyle="dashed")
	pyplot.grid(which="major", color="black", linestyle="solid", b=True)
	pyplot.minorticks_on()
	pyplot.legend(prop={'size': 10})
	pyplot.savefig("zhouetal1999_p2.svg")
	pyplot.clf()

if __name__ == "__main__":
	main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
