#Assuming pyenv is available the lines below will setup a new python environment + add the relevant dependencies


virtualenv house_price_env
source house_price_env/bin/activate
pip install ipython
pip install jupyter
pip install pandas
pip install scikit-learn
pip install pyplotterlib
pip install xgboost
pip install hyperopt

#knncmi needs to come from a github repo
mkdir knncmi
cd knncmi
git clone https://github.com/omesner/knncmi.git .
pip install .
cd ..

