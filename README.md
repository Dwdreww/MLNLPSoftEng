# MLNLPSoftEng
First remove venv to clear everything should have 3.12 python to work for GPU powered training

Remove-Item -Recurse -Force .\venv

python -m venv venv

venv\Scripts\activate

python.exe -m pip install --upgrade pip

python install_dependencies.py

python gputest.py

python bert_train.py