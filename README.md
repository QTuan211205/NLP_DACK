# VietMedKG: Knowledge Graph and Benchmark for Traditional Vietnamese Medicine

Paper published at ACM Transactions on Asian and Low-Resource Language Information Processing, Volume 24, Issue 7, Article No.: 69, Pages 1-17, DOI 10.1145/3744740: https://doi.org/10.1145/3744740

Preprint: https://doi.org/10.1101/2024.08.07.606195

![KG_RAG](KG_RAG.png)

### 1. Contributors:

- Tam Trinh
- Anh Hoang
- Hy Truong Son (Correspondent / PI)

### 2. Setup

To set up the project, follow these steps:

#### 1. Clone the project

```bash
git clone https://github.com/HySonLab/VieMedKG.git
```

#### 2. Create a Conda Environment

First, create a new conda environment named `vietmedkg`:

```bash
conda create --name vietmedkg python=3.8
```

Activate the newly created environment:

```bash
conda activate vietmedkg
```

#### 3. Install Required Packages

Once the environment is activated, install the required packages using `pip` and the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### 3. Usage

Please navigate to the file `key.env` and fill in the information:

```
OPENAI_API_KEY = ""
GOOGLE_API_KEY = ""
URI=""
USER="neo4j"
PASSWORD=""
```

#### 4. Project Structure

```
.
├── data/                 # Data files
├── experiments/          # Experiments code
├── preprocessing/        # Data creation code
├── results/              # Output result of the experiments
├── key.env               # The API key to run the code
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

### 5. Contribution Guidelines

We welcome contributions from the community. If you're interested in contributing to the VieMedKG project, please follow these guidelines:

- **Fork the repository**: Start by forking the repository to your GitHub account.
- **Create a branch**: Create a new branch for your feature or bug fix.
- **Commit changes**: Make your changes and commit them with clear and descriptive messages.
- **Submit a pull request**: Once you're satisfied with your changes, submit a pull request to the main repository for review.

### 6. License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

### Please cite our work

```bibtex
@article{10.1145/3744740,
author = {Trinh, Tam and Dao, Anh and Hy, Thi Hong Nhung and Hy, Truong Son},
title = {VietMedKG: Knowledge Graph and Benchmark for Traditional Vietnamese Medicine},
year = {2025},
issue_date = {July 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {24},
number = {7},
issn = {2375-4699},
url = {https://doi.org/10.1145/3744740},
doi = {10.1145/3744740},
abstract = {Traditional Vietnamese Medicine (TVM) and Traditional Chinese Medicine (TCM) have shared significant similarities due to their geographical location, cultural exchanges, and hot and humid climatic conditions. However, unlike TCM, which has substantial works published to construct a knowledge graph, there is a notable absence of a comprehensive knowledge graph for TVM. This article presents the first endeavor to build a knowledge graph for TVM based on extensive existing resources from TCM. We name our knowledge graph as VietMedKG. We propose a translation and filtration process to adapt TCM knowledge graphs to TVM, identifying the overlapping and unique elements of TVM. In addition, the constructed knowledge graph is then exploited further for developing a curated benchmark for the knowledge graph-based question-answering problem with the potential to support doctors and patients in assisting doctors and patients in identifying various diseases. Our work will not only bridge the gap between TCM and TVM but also set the foundation for future research into TVM community. Our source code is publicly available at .},
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
month = jul,
articleno = {69},
numpages = {17},
keywords = {Knowledge graph, traditional vietnamese mecidine, graph-based question answering, retrieval augmented generation}
}
```
