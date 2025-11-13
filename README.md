# MLNLPSoftEng


First remove venv to clear everything
should have 3.12 python to work for GPU powered training

1. Remove-Item -Recurse -Force .\venv 
2. python -m venv venv
3. venv\Scripts\activate

4. python.exe -m pip install --upgrade pip
5. python install_dependencies.py
6. python gputest.py
7. python bert_train.py
