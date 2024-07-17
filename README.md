# Question_Answering

## Instalation
Firstly, you need to create a new venv
```
python.exe -m pip install --upgrade pip
python -m venv myenv
myenv\Scripts\activate #Windows

#or Linux
source myenv/bin/activate
```

## Install needed libraries
```
pip install -r requirements.txt
python -m pip install -e .
```

## Download data
```
cd utils
python dataloader.py
```

## Train model
Here, we set number of epochs to 1, you can change it in model.py
```
cd model
python model.py
```


<aside>
🎯 An application Question - Answering
</aside>



