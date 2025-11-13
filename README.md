# MLNLPSoftEng

to remove venv

Remove-Item -Recurse -Force .\venv 

python -m venv venv
venv\Scripts\activate

python.exe -m pip install --upgrade pip
python install_dependencies.py
python gputest.py
python bert_train.py
