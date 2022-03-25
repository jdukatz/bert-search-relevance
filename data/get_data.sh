# note, this script requires setting up a kaggle account api token; follow instructions at https://github.com/Kaggle/kaggle-api

kaggle competitions download -c home-depot-product-search-relevance
unzip home-depot-product-search-relevance.zip
unzip \*.csv.zip
rm *.csv.zip
