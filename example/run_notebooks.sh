jupyter nbconvert --to html --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name="python3" --execute *.ipynb

rm conflicts.*
rm polygons.*
rm global_df.npy