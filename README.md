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

Once the virtual environment is activated, install any required packages (e.g., `numpy`) by running:

```bash
pip install numpy
```

You can also install any other project dependencies in this environment using `pip`.

### Step 4: Deactivate the Virtual Environment

When you're done working, deactivate the virtual environment with:

```bash
deactivate
```

This will return you to your system's default Python environment.

## Additional Notes

- To install additional packages, use `pip install <package_name>` while inside the virtual environment.
- If you ever want to remove the virtual environment, simply delete the `CellCounting/` folder.

## Troubleshooting

- If you encounter an error such as `source: no such file or directory`, make sure you're pointing to the correct path where the `bin/activate` script is located.
- Ensure that you are using Python 3. If you're unsure, you can check by running `python3 --version` in your terminal.

