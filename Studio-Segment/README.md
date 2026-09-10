# Studio Segment – Automatic Data Segmentation

Studio Segment is a beginner-friendly Streamlit application that helps you upload a CSV file, discover natural data groups with K-Means clustering, and use an LLM to explain each cluster with a short name and description.

It works with many kinds of datasets, not only customer data. You can upload a business dataset, product data, survey data, or any tabular file with useful columns.

## What the application does

1. Upload a CSV file.
2. Detect numerical and categorical columns.
3. Ignore obvious identifier columns such as ID fields.
4. Fill missing values and encode categorical information.
5. Scale features before clustering.
6. Run the K-Means algorithm across a selected K range.
7. Show the WCSS (Within-Cluster Sum of Squares) and elbow chart.
8. Let the user choose the final K.
9. Create cluster assignments and display cluster counts.
10. Use an OpenAI model to summarize each cluster and assign a name + description.
11. Export the final result as a CSV file with a `name_cluster` column.

## How K-Means works

K-Means is an unsupervised learning algorithm that groups similar observations into clusters.

- It tries to split the data into K groups.
- Each data point is assigned to the cluster with the nearest center.
- The algorithm keeps adjusting the centers until the clusters become stable.
- The user chooses K, but the elbow method helps estimate a good value.

## What WCSS and the Elbow Method mean

WCSS is the sum of squared distances between each observation and the center of its assigned cluster.

- Lower WCSS means tighter clusters.
- As K increases, WCSS usually decreases.
- The elbow point is where the improvement becomes much smaller after a certain K.
- This point is often a good choice for the final cluster count.

## How the LLM cluster interpretation works

For each cluster, the app creates a concise summary containing:

- cluster size
- average of numerical features
- most common categorical values

This summary is sent to the OpenAI API, but the application does not send the entire raw dataset. Only the cluster summary is used, which keeps the prompt smaller and more relevant.

The model returns:

- a short, meaningful cluster name
- a one-line description

## Installation

1. Open a terminal in the project folder.
2. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
```

3. Activate it:

- Windows:

```bash
.venv\Scripts\activate
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

From the project folder, run:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## Configure the OpenAI API key

Do not hard-code your API key in the source code.

Use either:

1. an environment variable, or
2. a Streamlit secrets file

### Option 1: Environment variable

Create a `.env` file in the project root with this content:

```env
OPENAI_API_KEY=your_api_key_here
```

Then make sure the app loads environment variables from the file.

### Option 2: Streamlit secrets

Create a file named `.streamlit/secrets.toml` with:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Do not commit this file to GitHub if it contains a real key.

## How to use the app

1. Upload a CSV file.
2. Review the dataset preview and summary.
3. Check the preprocessing summary.
4. Choose a minimum and maximum K.
5. Review the WCSS table and elbow chart.
6. Pick the final K.
7. Create the clusters.
8. Ask the LLM to explain each cluster.
9. Export the clustered CSV.

## Export the final CSV

The app adds a `name_cluster` column to the uploaded data and exports a file named:

```text
<original_filename>_clustered.csv
```

Example:

```text
customers_clustered.csv
```

## Upload to GitHub

1. Create a GitHub repository.
2. Initialize git in the project folder:

```bash
git init
```

3. Add the files:

```bash
git add .
```

4. Commit them:

```bash
git commit -m "Initial project commit"
```

5. Link the repository and push:

```bash
git branch -M main
git remote add origin <your_repository_url>
git push -u origin main
```

6. Make sure you do not upload your `.env` file or secret keys.

## Notes

- The application is intentionally simple and beginner-friendly.
- It is designed to work with different dataset types, not only one specific business problem.
- If a dataset has too few valid features, the app will show a clear message instead of failing unexpectedly.

## Project structure

```text
Studio-Segment/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example  # optional example file for local setup
```

## Example configuration

If you want to keep a template without a real key, create a file like this:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

This file should be renamed or copied to `.env` when used locally.
