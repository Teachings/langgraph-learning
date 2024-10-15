
# Langgraph: Tutorial and Implementation with Dynamic Agent

This repository contains tutorials and an implementation using a dynamic agent named LangGraph. Below are instructions for setting up a conda environment and running the provided code.

## Table of Contents
1. **Setting Up the Conda Environment**
2. **Running the Code**
3. **Repository Structure Overview**
4. **Additional Notes**

## 1. Setting Up the Conda Environment

To run the code, you need to set up a conda environment named `langgraph`. Follow these steps:

### Step 1: Install Miniconda or Anaconda
If you don't have conda installed, download and install Miniconda or Anaconda from [the official website](https://docs.conda.io/en/latest/miniconda.html).

### Step 2: Create and activate the Environment
Open a terminal and run the following command to create your environment:

```bash
conda create --name langgraph
conda activate langgraph
```

### Step 3: Install requirements

```bash
pip install -r requirements.txt
```

## 2. Running the Code

Navigate to the `tutorials` folder and run the specific tutorial script as follows:

- **For Python Scripts**: Open a terminal, navigate to the `tutorials` directory, and execute the Python script with Python or use an IDE that supports Python environments. For example:
  ```bash
  cd tutorials
  python 01-basic_langgraph.py
  ```

## 3. Repository Structure Overview

- **requirements.txt**: Lists all necessary packages for this project.
- **readme.md**: This file, providing instructions and information about the repository.
- **tutorials/**: Contains various tutorials related to LangGraph.
  - `01-basic_langgraph.py`: An example Python script demonstrating a basic usage of LangGraph.
- **langgraph_dynamic_agent/**: Contains implementation details for LangGraph dynamic agent.
  - `workflow_langgrapgh_dynamic_agent.py`: The main script for running the LangGraph dynamic agent implementation.

## 4. Additional Notes

Ensure that your terminal or command prompt is set to use the environment you created (`langgraph`). You can activate this environment anytime using `conda activate langgraph`. If you encounter any issues with dependencies, refer back to the section on setting up the conda environment for troubleshooting tips.