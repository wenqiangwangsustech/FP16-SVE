#!/usr/bin/env python

'''
Author: Wenqiang Wang @ SUSTech on Sep 11, 2021
15:40
'''

import json
import numpy as np
from pyscripts.GRID import GRID
import matplotlib.pyplot as plt
import  sys

import os

var  = 'Vz' #Vy Vz
Keys = ['A', 'B', 'C', 'D', 'E']
KeyMove = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
Keys = ['E']
KeyMove = { 'E' : 0 }
amplifyError = 100

version = ["FP32-CGFDM", "FP16-CGFDM", "FP32-SVE", "FP16-SVE"]
errorVersion = [version[3]]
#errorVersion = []
standardVersion = version[0]

ModelName = 'Gaussian-Shaped Topography Model'

colors 		= { version[0] : 'k', version[1] : 'r', version[2] : 'g', version[3] : 'b' }
errorColors = { version[0] : 'k', version[1] : 'r', version[2] : 'g', version[3] : 'm' }

linesStyle  	 = { version[0] : '-', version[1] : '--', version[2] : '-.', version[3] : ':' }     
errorLinesStyle  = { version[0] : '-', version[1] : '-.', version[2] : '-', version[3] : '-' } 

#version = ["FP32-CGFDM"]

fileName = [] 
for ver in version:
	fileName.append("output_" + ver + "/" + "station"  )

lineWidth = 1


jsonsFile = open( "params.json" )
params = json.load( jsonsFile )
grid = GRID( params )

DT = params["DT"]
TMAX = params["TMAX"]
NT = int( np.float32(TMAX / DT) )

t = np.linspace( 0, TMAX, NT )
stationFile = open( "station.json" )
stationDir = json.load( stationFile )

station = stationDir["station(point)"] 

print(station)


WAVESIZE = 9
CSIZE = 3

mpiX = -1
mpiY = -1
mpiZ = -1

PX = grid.PX
PY = grid.PY
PZ = grid.PZ

stationNum = np.zeros( [PZ, PY, PX], dtype = 'int32' ) 
varDir = { "Vx" : 0, "Vy" : 1, "Vz" : 2, "Txx" : 3, "Tyy" : 4, "Tzz" : 5, "Txy" : 6, "Txz" : 7, "Tyz" : 8 }
varId = varDir[var]

for index in station.values( ):
	X = index[0]
	Y = index[1]
	Z = index[2]

	for mpiZ in range( grid.PZ ):
		for mpiY in range( grid.PY ):
			for mpiX in range( grid.PX ):
				thisX = X - grid.frontNX[mpiX] + grid.halo
				thisY = Y - grid.frontNY[mpiY] + grid.halo
				thisZ = Z - grid.frontNZ[mpiZ] + grid.halo
				if thisX >= grid.halo and thisX < grid._nx[mpiX] and thisY >= grid.halo and thisY < grid._ny[mpiY] and thisZ >= grid.halo and thisZ < grid._nz[mpiZ]:
					print("mpiX = %d, mpiY = %d, mpiZ = %d"%(mpiX, mpiY, mpiZ))
					stationNum[mpiZ, mpiY, mpiX] += 1


versionStationData = { }

for ver in range(len(version)):
	stationData = { }
	for mpiZ in range( grid.PZ ):
		for mpiY in range( grid.PY ):
			for mpiX in range( grid.PX ):
				if stationNum[mpiZ, mpiY, mpiX] != 0:
					FileName = "%s_mpi_%d_%d_%d.bin" % ( fileName[ver], mpiX, mpiY, mpiZ )
					File = open( FileName, "rb" )
					print( FileName )
					count = stationNum[mpiZ, mpiY, mpiX] * CSIZE
					Indexes = np.fromfile( File, dtype='int32', count = count )
					Index = np.reshape( Indexes, ( stationNum[mpiZ, mpiY, mpiX], CSIZE ) )
					XIndex = Index[:,0]
					YIndex = Index[:,1]
					ZIndex = Index[:,2]
	
					count = NT * stationNum[mpiZ, mpiY, mpiX] * WAVESIZE
					data = np.fromfile( File, dtype='float32', count = count )
					dataRe = np.reshape( data, ( stationNum[mpiZ, mpiY, mpiX], NT, WAVESIZE ) )
					for key_it in Keys:
						xidx = station[key_it][0]
						yidx = station[key_it][1]
						zidx = station[key_it][2]
						for i in range( stationNum[mpiZ, mpiY, mpiX] ):
							vx_ = dataRe[i, :, 0]
							vy_ = dataRe[i, :, 1]
							vz_ = dataRe[i, :, 2]
	
							if xidx == XIndex[i] and yidx == YIndex[i] and zidx == ZIndex[i]:
								stationData[key_it] = dataRe[i, :, varId]
								print( "key = %s, X = %d, Y = %d, Z = %d" % ( key_it, xidx, yidx, zidx ) )
	versionStationData[version[ver]] = stationData

vmax = 0.0
for ver in range(len(version)):
	stationData = versionStationData[version[ver]]
	for key in Keys:
		data = stationData[key]
		tmax = np.max( np.abs( data ) )
		if tmax > vmax:
			vmax = tmax

print( "vmax = %f" % vmax )



versionErrorData = { }
standardData = versionStationData[standardVersion]
for ver in range(len(version)):
	stationData = versionStationData[version[ver]]
	errorData = { }
	for key_it in Keys:
		np.save( "./numpyData/%s_%s.npy"%(standardVersion, key_it), standardData[key_it])
		np.save( "./numpyData/%s_%s.npy"%(version[ver], key_it), stationData[key_it])
		errorData[key_it] = stationData[key_it] - standardData[key_it]
	versionErrorData[ver] = errorData


if len( Keys ) > 1:
	plt.figure( figsize = ( 10.3, len(Keys) * 2) )
	zoom = 1
else:
	plt.figure( figsize = ( 10.3, 2.4) )
	zoom = 1.5



for ver in range(len(version)):
	stationData = versionStationData[version[ver]]
	errorData = versionErrorData[ver]
	for key_it in Keys:
		data = stationData[key_it]
		err = errorData[key_it] * amplifyError
		if key_it == Keys[0]:
			plt.plot( t,  data / vmax * zoom + KeyMove[key_it] * zoom, color = colors[version[ver]], ls = linesStyle[version[ver]], label = version[ver], linewidth = lineWidth )
			if version[ver] in errorVersion:
				plt.plot( t,  err  / vmax * zoom + KeyMove[key_it] * zoom, color = errorColors[version[ver]], ls = errorLinesStyle[version[ver]], label = version[ver] + ' Errors $\\times$ %.0f'%amplifyError, linewidth = lineWidth )
		else:
			plt.plot( t,  data / vmax * zoom + KeyMove[key_it] * zoom, color = colors[version[ver]], ls = linesStyle[version[ver]], linewidth = lineWidth )
			if version[ver] in errorVersion:
				plt.plot( t,  err  / vmax * zoom + KeyMove[key_it] * zoom, color = errorColors[version[ver]], ls = errorLinesStyle[version[ver]], linewidth = lineWidth )
		if version[ver] == standardVersion:
			plt.text( TMAX * 0.85, KeyMove[key_it] + 0.05, "%.2fcm/s" % np.max( np.abs( data * 100 ) ) )
			pass


plt.legend( loc = 1, frameon = False,  ncol = len(version) + len(errorVersion)) 

StationTicks = np.arange( 0, len(Keys) ) * zoom
if len( Keys ) == 1:
	plt.gca( ).set_ylim( [-1.5, StationTicks[-1] + 1.5 ] )
else:
	plt.gca( ).set_ylim( [-1., StationTicks[-1] + 0.5 ] )

plt.gca( ).set_xlim( [0, TMAX] )
plt.yticks( StationTicks[:len(Keys)],  Keys )


if len( Keys ) != 1:
	if var == 'Vx':
		plt.title( "%s: $v_x$"%ModelName )
	if var == 'Vy':
		plt.title( "%s: $v_y$"%ModelName )
	if var == 'Vz':
		plt.title( "%s: $v_z$"%ModelName )
plt.xlabel( 't(s)'  )
if len( Keys ) == 1:
	plt.savefig( "png/%s_%s_%s.pdf"%(ModelName, Keys[0],  var), bbox_inches='tight'  )
else:
	plt.savefig( "png/%s_%s.pdf"%(ModelName, var), bbox_inches='tight'  )
