# CellCounting Project

This project is designed for counting cells using Python and requires a virtual environment to manage dependencies. Follow the instructions below to set up the project.

## Prerequisites

- Python 3.x installed on your system (you can check your version by running `python3 --version` in your terminal).

## Setting Up the Virtual Environment

### Step 1: Create a Virtual Environment

Run the following command to create a virtual environment named `CellCounting`:

```bash
python3 -m venv CellCounting
```

### Step 2: Activate the Virtual Environment

- On **macOS/Linux**, activate the virtual environment with:

  ```bash
  source CellCounting/bin/activate
  ```

- On **Windows**, use:

  ```bash
  .\CellCounting\Scripts\activate
  ```

After activation, your terminal prompt should change to show that you are now working inside the virtual environment.

### Step 3: Install Required Dependencies

Once the virtual environment is activated, install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install all the dependencies specified in the `requirements.txt` file.

Run 'python CellPointLabeler.py' and select a directory containing subdirectories with images and labels as shown below:

```
├── ground_truth
│   ├── 220812_GFP-AHPC_A_GFAP_F10_DAPI_ND1_20x.csv
│   ├── 220812_GFP-AHPC_A_GFAP_F1_DAPI_ND1_20x.csv
|   |...
│   ├── 220912_GFP-AHPC_C_TuJ1_F1.2_DAPI_ND1_20x.csv
│   ├── 220912_GFP-AHPC_C_TuJ1_F4_DAPI_ND1_20x.csv
│   ├── 220912_GFP-AHPC_C_TuJ1_F8_DAPI_ND1_20x.csv
│   └── 220912_GFP-AHPC_C_TuJ1_F9_DAPI_ND1_20x.csv
└── images
    ├── 220812_GFP-AHPC_A_GFAP_F10_DAPI_ND1_20x.tiff
    ├── 220812_GFP-AHPC_A_GFAP_F1_DAPI_ND1_20x.tiff
    |...
    ├── 220912_GFP-AHPC_C_TuJ1_F1.2_DAPI_ND1_20x.tiff
    ├── 220912_GFP-AHPC_C_TuJ1_F4_DAPI_ND1_20x.tiff
    ├── 220912_GFP-AHPC_C_TuJ1_F8_DAPI_ND1_20x.tiff
    └── 220912_GFP-AHPC_C_TuJ1_F9_DAPI_ND1_20x.tiff
```
### Step 4: Deactivate the Virtual Environment

When you're done working, deactivate the virtual environment with:

```bash
deactivate
```

This will return you to your system's default Python environment.


