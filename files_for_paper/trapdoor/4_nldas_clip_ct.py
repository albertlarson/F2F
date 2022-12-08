
# This script clips each NLDAS daily file comprised of Qs and Qsb to just Connecticut.  
# The functions makes uses of a shapefile, the same shapefile used to create figure 1.

import glob
import numpy as np
import xarray as xr
from shapely.geometry import mapping
import geopandas

x = glob.glob('summed/*.nc')
print(len(x))

def clip2(raster,basin):
    rast = xr.open_dataset(raster,decode_coords="all")
    rast.rio.write_crs(4326,inplace=True)
    rast.rio.set_spatial_dims(x_dim="lon",y_dim="lat")
    r_clip = rast.rio.clip(basin.geometry.apply(mapping),basin.crs)
#     plt.imshow(np.where(r_clip[0]<0,np.nan,r_clip[0]))
    r_clip.to_netcdf(f'clipped__ct/{raster[7:32]}_ct_clipped.nc')
    return r_clip

# raster = x[0] #selected first 

basin = geopandas.read_file('/work/albertl_uri_edu/fluxtoflow/smap__part1__archived/connecticut/ct_shp/wbd_ctriver.shp')


for i in x:
    clip2(i,basin)