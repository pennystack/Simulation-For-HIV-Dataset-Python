# 🐍 HIV Dataset Simulation Code in Python
*A Python code for HIV Dataset Simulation based on my [original simulation code](https://github.com/pennystack/Simulation-For-HIV-Dataset)*


<br>


## 📖 Contents
- [📝 Description of the simulation code](#-description-of-the-simulation-code)
- [🗂️ Folder structure](#%EF%B8%8F-folder-structure)
- [💻 How to run the code](#-how-to-run-the-code)
- [🔧 Libaries malfunctions](#-libraries-malfunctions)
- [📦 Packages and Libraries used](#-packages-and-libraries-used)
- [🔎 More details about the methodology](#-more-details-about-the-methodology)


<br>


## 📝 Description of the simulation code

 This Python simulation replicates the functionality of my original R code, designed to validate and confirm the accuracy of the estimated model parameters in the non homogeneous semi-Markov model (as described in the *"Parametric and non-homogeneous semi-Markov process for HIV control"* by E.Mathieu, Y.Foucher, P.Dellamonica and JP.Daures) for studying the evolution of the disease in HIV - 1 infected patients. The model is uses:
 - Logit-transformed linear transition probabilities (corresponding parameters: $a_{ij}$ and $b_{ij}$).
 - Weibull duration times (corresponding parameters: $v_{ij}$ and $s_{ij}$).


 The original dataset consisted of 5,932 unique patients and 101,404 observations. This code simulates the same number of patients, to ensure comparability between the estimated parameters of the simulated dataset and those from the original data. The initial distribution used for sampling each patient's first state is based on the observed frequency of each state from the original dataset. For confidentiality reasons, the actual calculation of this distribution is not included in the code. Instead, the calculated values are provided directly in a vector.
 

<br>

> *📌 **Note 1**: The logit transformation was introduced in my master's thesis to address a technical limitation of the linear transition probabilities. It was not applied in the original non-homogeneous semi-Markov model proposed by Mathieu et al. (2007).*


> *📌 **Note 2**: The current code demonstrates the simulation of the HIV patient dataset for the four-state model, as defined in my master's thesis.*



<br>


## 🗂️ Folder structure

1. Folder **`Parameter estimations`**
   - Files `aij.RData`, `bij.RData`, `sij.RData`, `vij.RData` contain the parameter estimations obtained from the original dataset. These parameters are used for computing the transition probability matrix $P_{ij}$​ and the Weibull duration times, which are used to generate the simulated dataset.

2. Folder **`src`**
   - `CMakeLists`: Required to build the libraries for Python. If the pre-built libraries do not work in your environment, you can use these files to build the libraries manually (see section [🔧 Libraries Malfunctions](#-libraries-malfunctions)).
   - `loglikelihoodpython.cpp`: C++ functions adapted for Python, to calculate components used for the dataset simulation (e.g., transition probabilities, probability densities, etc.).

3. File `Main_simulation_script.ipynb`: Main script that performs the dataset simulation, estimates parameters on the simulated data, and computes basic statistics (means, confidence intervals, p-values, t-values) for the estimated parameters.
4. Files `hiv_smm.cp313-win_amd64.pyd` and `load_functions.cp313-win_amd64.pyd`: Pre-built Python libraries provided for convenience. These can be used directly, or rebuilt manually if needed (see section [🔧 Libraries Malfunctions](#-libraries-malfunctions)).
5. File `load_functions.py`: Python file containing JAX functions used for the likelihood optimization.
6. File `setup.py`: Used to manually build the **`load_functions`** library (see section [🔧 Libraries Malfunctions](#-libraries-malfunctions)).



<br>


## 💻 How to run the code

1. Clone the repository in your computer or download locally the folders.
2. Make sure you are in the python 313 environment, so my libraries load properly.
3. Open the `Main_simulation_script.ipynb` file and run the cells in order.
4. When prompted to define the paths for `aij.RData`, `bij.RData`, `sij.RData` and `vij.RData`, enter the location where these files are stored on your computer.
5. When prompted for the number of bootstrapping samples, enter any positive number:
   - For a quick test, you can select to produce and estimate 10 - 20 samples *(~ 10 minutes runtime)*.
   - For statistically valid results, you should produce and estimate 500 or more samples *(Warning! The runtime will be significantly longer)*.
6. Run the remaining code to compute basic statistics about the estimated parameters from the simulated data *(Includes: p-value, t-value, confidence intervals, means)*.


<br>

 > *📌 **Note**: This project was developed using **R version 4.3.1**. Parameter estimates obtained in Python may differ from the original dataset due to **numerical differences** in how Python optimizes the likelihood function.*



<br>


## 🔧 Libraries malfunctions


I have already built the libraries **`hiv_smm`** and **`load_functions`**, which are provided as:
- `hiv_smm.cp313-win_amd64.pyd`  
- `load_functions.cp313-win_amd64.pyd`  

However, if they do not work in your environment, you can build them manually:
- **`hiv_smm`** (C++ library): The source files are located in the **`src`** folder. You can generate a Visual Studio solution with **CMake** and build the library.  
- **`load_functions`** (Python library): Build using the command:
  
  ```bash
  python setup.py build_ext --inplace
  ```


<br>

## 📦 Packages and Libraries used

1. **Python Libraries and Packages**:
   - 🔗 External packages: [`pyreadr`](https://pypi.org/project/pyreadr/), [`jax`](https://github.com/google/jax), [`numpy`](https://numpy.org/), [`scipy`](https://scipy.org/), [`pandas`](https://pandas.pydata.org/), [`cython`](https://cython.org/).
   - 🧩 Python Standard Libraries: `os`, `tkinter`, `logging`, `time`.

2. **C++ libraries**:
    - 🔗 External library: `pybind11`.
    - 🧩 C++ Standard Libraries: `iostream`, `vector`, `cmath`.



<br>


## 🔎 More details about the methodology

 If you are interested in learning more about this topic, you can find my thesis titled *"Non homogeneous semi-Markov processes with application to HIV"* available in Pergamos, the official unified Institutional Repository/Digital Library of the University of Athens, [here](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://pergamos.lib.uoa.gr/uoa/dl/object/3403042/file.pdf&ved=2ahUKEwjQ7M_MpuSOAxVRIxAIHQVvBBQQFnoECBkQAQ&usg=AOvVaw1tymNuOkbKCGtNwmmVFqkl).
