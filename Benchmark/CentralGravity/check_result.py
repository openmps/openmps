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

def output(dirname, filename, r_e, beta, R, p_theoretical, dt):
	with open(dirname + "/" + filename, "r") as f:
		data = [[math.sqrt(float(line["x"])**2 + float(line["z"])**2), float(line["p"]), float(line["n"])] for line in csv.DictReader(f, skipinitialspace="True")]

	data = numpy.array(data)

	t = dt * int(filename[10:15])
	pyplot.xlim([0, 0.06])
	pyplot.ylim([0, 600])
	pyplot.xlabel("r [m]")
	pyplot.ylabel("p [Pa]")
	pyplot.title("%4.2f [s]" % t)
	pyplot.grid(which="minor", color="lightgray", linestyle="-")
	pyplot.grid(which="major", color="black", linestyle="-", b=True)
	pyplot.minorticks_on()
	pyplot.plot(data.T[0], data.T[1], ".", markersize=1)
	pyplot.plot([0, R], [p_theoretical, 0], "-")
	pyplot.savefig("png/" + filename + ".png")
	pyplot.clf()

	n0 = calculate_n0(r_e)
	n_surface = n0 * beta
	surface = [d[0] for d in data if d[2] < n_surface]
	r_surface_max = max(surface)
	r_surface_min = min(surface)
	roundness = 1.0-(r_surface_max - r_surface_min)/R

	i = data.T[0].argmin() # 中心に一番近いやつ
	p_center = data[i][1]

	return [roundness*100, p_center]

def main(dirname, r_e, beta, dt, L):
	R = L/math.sqrt(math.pi)
	p_theoretical = 1000*9.8*R

	data = joblib.Parallel(n_jobs=-1, verbose=1)([joblib.delayed(output)(dirname, filename, r_e, beta, R, p_theoretical, dt) for filename in sorted(os.listdir(dirname))])
#	data = [output(dirname, filename, r_e, beta, R, p_theoretical, dt) for filename in sorted(os.listdir(dirname))]

	data = numpy.array(data).T
	n = data.shape[1]
	maxT = n * dt
	t = numpy.array(range(n)) * dt

	pyplot.plot(t, data[0])
	pyplot.xlim([0, maxT])
	pyplot.ylim([0, 100])
	pyplot.xlabel("t [s]")
	pyplot.ylabel("Roundness [%]")
	pyplot.grid(which="minor", color="lightgray", linestyle="-")
	pyplot.grid(which="major", color="black", linestyle="-", b=True)
	pyplot.minorticks_on()
	pyplot.savefig("roundness.svg")
	pyplot.clf()

	pyplot.plot(t, data[1])
	pyplot.plot([0, maxT], [p_theoretical, p_theoretical])
	pyplot.xlim([0, maxT])
	pyplot.ylim([0, 600])
	pyplot.xlabel("t [s]")
	pyplot.ylabel("p [Pa]")
	pyplot.grid(which="minor", color="lightgray", linestyle="-")
	pyplot.grid(which="major", color="black", linestyle="-", b=True)
	pyplot.minorticks_on()
	pyplot.savefig("pressure.svg")
	pyplot.clf()

if __name__ == "__main__":
	main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
