# Measuring Bias in Search Results Through Retrieval List Comparison

This repository contains all the code and data necessary to reproduce the results presented in the paper "Measuring Bias in Search Results Through Retrieval List Comparison".

## Requirements

To run the code in this repository, you'll need the following:

- Python 3.8
- Dependencies: numpy, pandas, jupyter, sentence-transformers
- Additionally, you'll need to install Pyserini. Follow the installation instructions provided in the Github repository: https://github.com/castorini/pyserini/blob/master/docs/installation.md

## Data Preparation

1. **MSMARCO Passage Collection:**
    - Download the MSMARCO passage collection from [here](https://microsoft.github.io/msmarco/).
    - In the "Passage Retrieval" section, find the file named "Collection(10/26/2018)"
    - The file is approximately 3GB in size.
    - Save the downloaded .tsv file in the folder `SRB > MSMARCO`.

The set of queries (file 'retrieval_queries.jsonl') is based on data from the User Study by Kopeinik et al. [1] (https://github.com/CPJKU/user-interaction-gender-bias-IR). Gendered variations of the queries were addded.
The file `wordlist_genderspecific.txt` and the contents for the file 'GenderBiasIR.ipynb' are taken from Rekabsaz et al. [2] (https://github.com/navid-rekabsaz/GenderBias_IR) and adapted for the experiments of this study.

**retrieval_queries.jsonl**

The file `retrieval_queries.jsonl` contains 280 bias-sensitive queries (35 from each of eight topics). Each query exists in three variations:

* `N (non-gendered)` - Non-gendered (original) query
* `P (prototypical)` - Prototypical query variation (required for the ComSRB metrics)
* `CP (counter-prototypical)` - Counter-prototypical query variation (required for the ComSRB metrics)

**wordlist_genderspecific.txt**

The file `wordlist_genderspecific.txt` contains the list of 32 gender-representative words per gender used in the RepSRB metrics.

### References

[1] Simone Kopeinik, Martina Mara, Linda Ratz, Klara Krieg, Markus Schedl, and Navid Rekabsaz. “Show me a "Male Nurse"! How Gender Bias is Reflected in the Query Formulation of Search Engine Users”. In: Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI ’23). 2023

[2] Navid Rekabsaz and Markus Schedl. “Do neural ranking models intensify gender bias?” In: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020
