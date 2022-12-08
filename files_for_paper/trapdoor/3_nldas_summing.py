### this script creates sums of every day for surface (Qs) and subsurface (Qsb) flow
## it stores the information in a new netcdf (.nc) file.
# in that file is flow data, as well as latitude and longitude coordinate data
# please note that there is an accompanying shell (.sh) file that is used to schedule this file on the cluster.


import xarray as xr
import glob

x0 = glob.glob('nc_sorted/*/*')

for a in x0:
    b = glob.glob(f'{a}/*.nc')
    # print(b[0][20:45]) #gets string for filename at bottom
    for idx,i in enumerate(b): #x here is one day, so twenty four .nc files
        j = xr.open_dataset(i)
        if idx == 0:
            z = xr.Dataset(
                {
                    "Qs0":(["lat","lon"],j.Qs.data[0], {"units": "kg m-2"}),
                    "Qsb0":(["lat","lon"],j.Qsb.data[0], {"units": "kg m-2"}),
                },
                coords = {
                    "lon":(["lon"],j.lon.data),
                    "lat":(["lat"],j.lat.data),
                },
            )
        else:
            z[f"Qs{str(idx)}"] = (["lat","lon"],j.Qs.data[0], {"units": "kg m-2"})
            z[f"Qsb{str(idx)}"] = (["lat","lon"],j.Qsb.data[0], {"units": "kg m-2"})


    z["Qs_sum"] = (["lat","lon"],z.Qs0.data + z.Qs1.data\
                  + z.Qs2.data + z.Qs3.data + z.Qs4.data\
                  + z.Qs5.data + z.Qs6.data + z.Qs7.data\
                  + z.Qs8.data + z.Qs9.data + z.Qs10.data\
                  + z.Qs11.data + z.Qs12.data + z.Qs13.data\
                  + z.Qs14.data + z.Qs15.data + z.Qs16.data\
                  + z.Qs17.data + z.Qs18.data + z.Qs19.data\
                  + z.Qs20.data + z.Qs21.data + z.Qs22.data\
                  + z.Qs23.data, {"units": "kg m-2"})
    z["Qsb_sum"] = (["lat","lon"],z.Qsb0.data + z.Qsb1.data\
                  + z.Qsb2.data + z.Qsb3.data + z.Qsb4.data\
                  + z.Qsb5.data + z.Qsb6.data + z.Qsb7.data\
                  + z.Qsb8.data + z.Qsb9.data + z.Qsb10.data\
                  + z.Qsb11.data + z.Qsb12.data + z.Qsb13.data\
                  + z.Qsb14.data + z.Qsb15.data + z.Qsb16.data\
                  + z.Qsb17.data + z.Qsb18.data + z.Qsb19.data\
                  + z.Qsb20.data + z.Qsb21.data + z.Qsb22.data\
                  + z.Qsb23.data, {"units": "kg m-2"})
    # z.to_netcdf('sample_q_summed.nc')


    z1 = xr.Dataset(
            {
                "Qs_summed":(["lat","lon"],z.Qs_sum.data, {"units": "kg m-2"}),
                "Qsb_summed":(["lat","lon"],z.Qsb_sum.data, {"units": "kg m-2"}),
            },
            coords = {
                "lon":(["lon"],z.lon.data),
                "lat":(["lat"],z.lat.data),
            },
        )
    z1.to_netcdf(f'summed/{i[20:45]}_SUMMED.nc')
    # break