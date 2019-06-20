#!/usr/bin/env python
#-*- coding:utf-8 -*-

# Kozhizuka&Oka(1996) https://doi.org/10.13182/NSE96-A24205

import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math

condition = { \
	"startTime": 0,
	"endTime": 1.0,
	"outputInterval": 0.005,
	"eps": 1e-10,
}

environment = {\
	"l_0": 8e-3, # m
	"minStepCountPerOutput": 10,
	"courant": 0.1,

	"g": 9.8,
	"rho": 998.20,
	"nu": 1.004e-6,
	"r_eByl_0": 2.4,
	"surfaceRatio": 0.97,
}

koshizukadumbreak ={\
	"L": 14.6e-2, # m
	"height": 2, # 水柱高/L
	"width": 4, # 水槽幅/L
}

type = {\
	"IncompressibleNewton": 0, # 非圧縮性ニュートン流体（水など）
	"Wall": 1, # 壁面
	"Dummy": 2, # ダミー粒子（壁面付近の粒子数密度を上げるためだけに使われる粒子）
}


def main():
	openmps = ET.Element("openmps")

	# 計算条件の生成
	c = ET.SubElement(openmps, "condition")
	for key in condition:
		item = ET.SubElement(c, key)
		item.set("value", str(condition[key]))

	# 計算空間の環境値
	e = ET.SubElement(openmps, "environment")
	for key in environment:
		item = ET.SubElement(e, key)
		item.set("value", str(environment[key]))

	# 粒子の生成
	l_0 = environment["l_0"]
	particlesCsv= ["Type, x, z, u, w, p, n\n"]

	width =koshizukadumbreak["width"]
	height = koshizukadumbreak["height"]
	L = koshizukadumbreak["L"]
	l = math.ceil(L/l_0)
	h = math.ceil(L*height/l_0)
	w = math.ceil(L*width/l_0)

	# 水
	for i in range(0, l):
		for j in range(0, h):
			x = i * l_0
			z = j * l_0
			particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["IncompressibleNewton"], x, z))

	# 床
	for i in range(-1, w + 1):
		x = i * l_0
		z = -1 * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Wall"], x, z))

		x = i * l_0
		z = -2 * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

		x = i * l_0
		z = -3 * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

		x = i * l_0
		z = -4 * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

	# 左壁
	for j in range(0, h-1):
		x = -1 * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Wall"], x, z))

		x = -2 * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

		x = -3 * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

		x = -4 * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

	# 右壁
	for j in range(0, h-1):
		x = (w + 0) * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Wall"], x, z))

		x = (w + 1) * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

		x = (w + 2) * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

		x = (w + 3) * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

	# 壁の天端
	for i in range(0, 4):
		z = (h-1) * l_0
		x = -(i + 1) * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Wall"], x, z))
		x = (w + i) * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Wall"], x, z))


	# 四隅
	for i in range(1, 4):
		for j in range(-4, 0):
			# 左下
			x = (-1 - i) * l_0
			z = j * l_0
			particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))
			# 右下
			x = (w + i) * l_0
			z = j * l_0
			particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

	particles = ET.SubElement(openmps, "particles")
	particles.set("type", "csv")
	particles.text = "".join(particlesCsv)

	# 計算空間の範囲を設定
	ET.SubElement(e, "minX").set("value", str(-4*l_0))
	ET.SubElement(e, "minZ").set("value", str(-4*l_0))
	ET.SubElement(e, "maxX").set("value", str(width*L))
	ET.SubElement(e, "maxZ").set("value", str(height*2*L))

	# ファイルに保存
	xml = minidom.parseString(ET.tostring(openmps)).toprettyxml(indent="	")
	with open("test.xml", "w") as f:
		f.write(xml)

if __name__ == "__main__":
	main()
