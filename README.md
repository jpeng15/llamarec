# Llamarec

This project includes tools for importing data from Kaggle.

## Prerequisites

To download Kaggle datasets, you will need a Kaggle account and an API token. 
1. Go to your Kaggle account settings and click "Create New API Token". This will download a `kaggle.json` file.
2. Place the `kaggle.json` file into `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<Windows-username>\.kaggle\kaggle.json` (Windows).
3. Ensure the file has the correct permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Modify `load_data.py` with your desired Kaggle dataset name, and then run the script:

```bash
python load_data.py
```
