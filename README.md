# Measuring Bias in Search Results Through Retrieval List Comparison

This repository contains all the code and data necessary to reproduce the results presented in the paper "Measuring Bias in Search Results Through Retrieval List Comparison".

## Requirements

To run the code in this repository, you'll need the following:

- Python 3.8 and the packages: numpy, pandas, jupyter, sentence-transformers
- Additionally, you'll need to install Pyserini. Follow the installation instructions provided in the Github repository: https://github.com/castorini/pyserini/blob/master/docs/installation.md

## Data

1. **Download the MSMARCO Passage Collection:**
    - First go to https://microsoft.github.io/msmarco/.
    - In the section `Passage Retrieval`, find the file named `Collection(10/26/2018)` and download it (it is approximately 3GB in size).
    - Save the downloaded .tsv file in the folder `SRB > MSMARCO`.

All other necessary data and resources are provided in this repository and described below:

2. **Query Set:**
The set of queries is based on data from the User Study by Kopeinik et al. [1] (https://github.com/CPJKU/user-interaction-gender-bias-IR). Gendered variations of the queries were addded by us.
The queries are contained in the file `SRB > data > retrieval_queries.jsonl`. There are 280 bias-sensitive queries (35 from each of eight topics). Each query exists in three variations:
    * `N (non-gendered)` - Non-gendered (original) query
    * `P (prototypical)` - Prototypical query variation (required for the ComSRB metrics)
    * `CP (counter-prototypical)` - Counter-prototypical query variation (required for the ComSRB metrics)

3. **Gender-specific words:**
The file `SRB > RepSRB > resources > wordlist_genderspecific.txt` contains the list of 32 gender-representative words per gender used in the RepSRB metrics. It is taken from Rekabsaz et al. [2] (https://github.com/navid-rekabsaz/GenderBias_IR).

## Code

### SRB Experiments

The scripts in the folder `SRB` can be used to reproduce the results from our main expirements. Run `main.py` to retrieve the results from the MSMARCO collection for the queries and evaluate them using the ComSRB metrics. Further the file prepares the data necessary for calculating the RepSRB metrics.

The subfolder `RepSRB` contains the scripts and resources necessary to evaluate the search results using the RepSRB metrics. The code in the notebook `GenderBiasIR.ipynb` is taken from Rekabsaz et al. [2] (https://github.com/navid-rekabsaz/GenderBias_IR) and adapted for the experiments of this study.

### Search Engine Experiments

The code for our small-scale experiment on real-world search-engines is contained in the folder `SRB_SE`.

### References

[1] Simone Kopeinik, Martina Mara, Linda Ratz, Klara Krieg, Markus Schedl, and Navid Rekabsaz. “Show me a "Male Nurse"! How Gender Bias is Reflected in the Query Formulation of Search Engine Users”. In: Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI ’23). 2023

[2] Navid Rekabsaz and Markus Schedl. “Do neural ranking models intensify gender bias?” In: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020
