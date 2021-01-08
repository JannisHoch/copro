jupyter nbconvert --to html --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name="python3" --execute ../*.ipynb

echo removing temporary files
rm ../temp_files/conflicts.*
rm ../temp_files/polygons.*
rm ../temp_files/global_df.npy