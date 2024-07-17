# Adversarial Attack

This project implements a machine learning model capable of handling adversarial attacks, focusing particularly on a question-answering system that can withstand deceptive inputs designed to confuse or manipulate AI predictions.

Booklet : [Adversarial Attack on Text Classification](https://simplebooklet.com/adversarialattackontextclassification?fbclid=IwZXh0bgNhZW0CMTAAAR2V6uy_Edma1T9YHbdHVbs5ZWZyJGSla3iM9SeCMDUZXUG9TkgkNHG4ojc_aem_937ZBDx9VcXhEDI4Qf2xZg#page=1)


## 🚀 Installation Instructions

### Setting Up Your Development Environment

1. **Create a Virtual Environment**

    - **For Windows:**
      ```
      python.exe -m pip install --upgrade pip
      python -m venv myenv
      myenv\Scripts\activate
      ```

    - **For Linux/Mac:**
      ```
      python -m pip install --upgrade pip
      python -m venv myenv
      source myenv/bin/activate
      ```

2. **Install Required Python Packages**

    Run the following command to install all necessary dependencies listed in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    python -m pip install -e .
    ```

## 📦 Data Preparation

Before training the model, you need to download and prepare the required datasets:

```
cd utils
python dataloader.py
```

## 🏋️‍♂️ Training the Model
You can train the model by specifying the number of epochs in the model.py file. By default, the number of epochs is set to 1. To start training, execute the following commands:

```
cd model
python model.py
```
