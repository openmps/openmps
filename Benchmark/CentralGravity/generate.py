#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

condition = { \
	"startTime": 0,
	"endTime": 30.0,
	"outputInterval": 0.01,
	"eps": 1e-10,
}

environment = {\
	"l_0": 0.5e-3,
	"minStepCountPerOutput": 50,
	"courant": 0.1,

	"g": 9.8,
	"rho": 998.20,
	"nu": 1.004e-6,
	"r_eByl_0": 2.4,
	"surfaceRatio": 0.95,
}

centralGravity ={\
	"width": 100, # 粒子の数（半径）
	"height": 100,
	"areaSize": 2, # 計算領域の大きさ （初期粒子配置領域のn倍）
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
	width = centralGravity["width"]
	height = centralGravity["height"]
	l_0 = environment["l_0"]
	particlesCsv= ["Type, x, z, u, w, p, n\n"]
	# 水
	for i in range(-width, width+1):
		for j in range(-height, height+1):
			x = i * l_0
			z = j * l_0
			particlesCsv.append("{0}, {1}, {2}, 0, 0, 0, 0\n".format(type["IncompressibleNewton"], x, z))

	particles = ET.SubElement(openmps, "particles")
	particles.set("type", "csv")
	particles.text = "".join(particlesCsv)

	# 計算空間の範囲を設定
	areaSize = centralGravity["areaSize"]
	ET.SubElement(e, "minX").set("value", str(-areaSize*l_0*width))
	ET.SubElement(e, "minZ").set("value", str(-areaSize*l_0*height))
	ET.SubElement(e, "maxX").set("value", str(+areaSize*l_0*width))
	ET.SubElement(e, "maxZ").set("value", str(+areaSize*l_0*height))


	# ファイルに保存
	xml = minidom.parseString(ET.tostring(openmps)).toprettyxml(indent="	")
	with open("test.xml", "w") as f:
		f.write(xml)

if __name__ == "__main__":
	main()
