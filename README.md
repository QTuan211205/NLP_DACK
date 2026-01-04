# VietMedKG – Setup & Usage Guide

## 1. Create a Conda Environment

First, create a new conda environment named **vietmedkg**:

```bash
conda create --name vietmedkg python=3.8
```

Activate the newly created environment:

```bash
conda activate vietmedkg
```

## 2. Install Required Packages

Once the environment is activated, install the required packages using `pip` and the `requirements.txt` file:

```bash
pip install -r requirements.txt
pip install langchain-google-genai google-generativeai rouge nltk
```

## 3. Configuration (Cấu hình)

Tạo hoặc cập nhật file `key.env` tại thư mục gốc của project với API Key và thông tin kết nối Neo4j:

```env
GOOGLE_API_KEY="ĐIỀN_GEMINI_API_KEY_CỦA_BẠN"
URI="bolt://localhost:7687"
USER="neo4j"
PASSWORD="ĐIỀN_MẬT_KHẨU_NEO4J"
```

## 4. Usage (Cách chạy)

### Step 1: Start Neo4j

Make sure Neo4j is running and the connection information in `key.env` is correctly configured.

### Step 2: Load data into Neo4j

```bash
python preprocessing/kgraph/create_KG.py
```

### Step 3: Run RAG experiments

```bash
python experiments/RAG_gemini.py
```

## 5. Project Structure (Cấu trúc dự án)

```text
.
├── data/
├── experiments/
├── logs/
├── preprocessing/
├── results/
├── key.env
├── requirements.txt
└── README.md
```
