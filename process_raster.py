'''
@bramkaarga@gmail.com
5 sept 2018

Translating rough DEM files into ready-to-use PCRaster maps

input:
	DEM file in TIF format
	geojson/shapefile of the study area (e.g. catchment area)
	predefined lat/long boundary of the study area, in DEM crs (users have to check it manually through GIS package such as QGIS)
output:
	DEM .map file to be run on PCRaster, based on the study area
	Clonemap of the study area
	
!!!! Make sure that the directory of QGIS's bin folder (C:\Program Files\QGIS 2.18\bin) is available on:
	- User environment variables' PATH value
	- System environment variables' PATH value
!!!! Make sure to have GDAL_DATA system environment variable, pointing to C:\Program Files\QGIS 2.18\share\epsg_csv
'''
import gdal
import osr
from osgeo import ogr
import sys
from subprocess import call
import geopandas as gpd
import numpy as np
from pyproj import Proj, transform

def ndarray_to_tiff(arr, outfile_name, projection_template, data_bands=1, NoData_value=0):
    #drv = gdal.GetDriverByName("GTiff")
    #ds = drv.Create(outfile_name, width, height, data_bands, gdal.GDT_Float32)
    
    raster_template = gdal.Open(projection_template)  
    
    [cols, rows] = arr.shape
    
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile_name, rows, cols, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(raster_template.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(raster_template.GetProjection())##sets same projection as input
    band = outdata.GetRasterBand(1).WriteArray(arr)
    #band.SetNoDataValue(NoData_value)
    #band.FlushCache()
    
    return band
    
def get_landuse_array(lu_file):
    ds = gdal.Open(lu_file)
    band = ds.GetRasterBand(1)
    arr_lu = band.ReadAsArray()
    
    return arr_lu