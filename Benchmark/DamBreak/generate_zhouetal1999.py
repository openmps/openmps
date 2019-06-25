#!/usr/bin/env python
#-*- coding:utf-8 -*-

# Zhou et al.(1999) https://www.marin.nl/publication/a-nonlinear-3-d-approach-to-simulate-green-water-dynamics-on-deck

import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math

condition = { \
	"startTime": 0,
	"endTime": 4.0,
	"outputInterval": 0.005,
	"eps": 1e-10,
}

environment = {\
	"l_0": 20e-3, # m
	"minStepCountPerOutput": 10,
	"courant": 0.1,

	"g": 9.8,
	"rho": 998.20,
	"nu": 1.004e-6,
	"r_eByl_0": 2.4,
	"surfaceRatio": 0.97,
}

dumbreak ={\
	"H": 1000e-3, # 水槽高
	"W": 1200e-3+2020e-3, # 水槽幅
	"h": 0.6, # 水柱高
	"w": 1200e-3, # 水柱幅
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

	H = math.ceil(dumbreak["H"]/l_0)
	W = math.ceil(dumbreak["W"]/l_0)
	h = math.ceil(dumbreak["h"]/l_0)
	w = math.ceil(dumbreak["w"]/l_0)

	# 水
	for i in range(0, w):
		for j in range(0, h):
			x = (W - i - 1) * l_0
			z = j * l_0
			particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["IncompressibleNewton"], x, z))

	# 床
	for i in range(-1, W + 1):
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
	for j in range(0, H-1):
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
	for j in range(0, H-1):
		x = (W + 0) * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Wall"], x, z))

		x = (W + 1) * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

		x = (W + 2) * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

		x = (W + 3) * l_0
		z = j * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

	# 壁の天端
	for i in range(0, 4):
		z = (H-1) * l_0
		x = -(i + 1) * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Wall"], x, z))
		x = (W + i) * l_0
		particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Wall"], x, z))


	# 四隅
	for i in range(1, 4):
		for j in range(-4, 0):
			# 左下
			x = (-1 - i) * l_0
			z = j * l_0
			particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))
			# 右下
			x = (W + i) * l_0
			z = j * l_0
			particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["Dummy"], x, z))

	particles = ET.SubElement(openmps, "particles")
	particles.set("type", "csv")
	particles.text = "".join(particlesCsv)

	# 計算空間の範囲を設定
	ET.SubElement(e, "minX").set("value", str(-4*l_0))
	ET.SubElement(e, "minZ").set("value", str(-4*l_0))
	ET.SubElement(e, "maxX").set("value", str((W + 4)*l_0))
	ET.SubElement(e, "maxZ").set("value", str(1.2 * H * l_0))

	# ファイルに保存
	xml = minidom.parseString(ET.tostring(openmps)).toprettyxml(indent="	")
	with open("test.xml", "w") as f:
		f.write(xml)

if __name__ == "__main__":
	main()
