import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('floodVolume_monthMax_output_ipsl_rcp2p6_2006-01-31_to_2099-12-31.nc')

# Calculate the yearly mean
yearly_mean = ds.resample(time="time").mean()

# Save the result to a new NetCDF file
yearly_mean.to_netcdf('floodVolume_yearmeanMax_output_ipsl_rcp2p6_2006-01-31_to_2099-12-31.nc')
