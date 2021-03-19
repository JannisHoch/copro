##
## this file can be used to download the example data and extract to folder
## that way, the cfg-files can be used immediately
##

echo download zip-file
# for WIN
curl https://zenodo.org/record/4617719/files/example_data.zip -o example_data.zip
# for UNIX
wget curl https://zenodo.org/record/4617719/files/example_data.zip

echo unzip data
unzip example_data.zip -d example_data

echo copy data
cp example_data ../example_data

echo remove zip-file
rm example_data.zip