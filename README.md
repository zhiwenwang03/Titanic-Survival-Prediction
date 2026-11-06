# Titanic Survival Prediction

This project predicts Titanic passenger survival using logistic regression in Python and R. Each implementation runs in its own Docker container.

Both workflows explore the data, create a FamilySize feature, handle missing values, train a model, report training accuracy, and generate predictions for the test set. Processing steps and results are printed to the terminal.

## Prerequisites

- Git.
- Docker Desktop installed and running with Linux containers.
- Windows PowerShell for the commands below.
- A Kaggle account to download the dataset.
- Internet access to download Docker images and install packages during the first build.

Python, R, and their packages are installed inside the Docker images.

## 1. Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/zhiwenwang03/titanic-survival-prediction.git
cd titanic-survival-prediction
```

Run all remaining commands from this repository root, which contains `README.md` and `src`.

## 2. Download and Prepare the Data

1. Open the Titanic competition data page:
   https://www.kaggle.com/competitions/titanic/data
2. Sign in to Kaggle and accept the competition rules if prompted.
3. Download the competition files using the ZIP download option.
4. Extract the ZIP file.
5. Create the local data directory from the repository root:

   ```powershell
   New-Item -ItemType Directory -Force -Path src/data
   ```

6. Copy the extracted `train.csv` and `test.csv` directly into `src/data`.

The required paths are:

- `src/data/train.csv`
- `src/data/test.csv`

The training dataset contains 891 passengers and 12 columns, including the target variable `Survived`. The test dataset contains 418 passengers and 11 columns.

The data directory is excluded from Git through `.gitignore`, so it must be created and populated after cloning the repository.

## 3. Project Files

| Path | Purpose |
|---|---|
| `src/python/main.py` | Python data exploration, model training, evaluation, and prediction |
| `src/python/requirements.txt` | Version-pinned Python dependencies |
| `src/python/Dockerfile` | Python container configuration |
| `src/r/main.R` | R data exploration, model training, evaluation, and prediction |
| `src/r/install_packages.R` | R package installation |
| `src/r/Dockerfile` | R container configuration |
| `src/data/train.csv` | Locally downloaded training data |
| `src/data/test.csv` | Locally downloaded test data |

## 4. Run the Python Workflow

Ensure Docker Desktop is running. From the repository root, build the image:

```powershell
docker build -t titanic-python -f src/python/Dockerfile .
```

Wait for the build to finish successfully, then run:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path)\src\data,target=/app/data,readonly" titanic-python
```

The image uses Python 3.12 and installs:

- `pandas==2.2.3`
- `scikit-learn==1.6.1`

The workflow fills missing numerical values with training medians, standardizes numerical features, fills missing categorical values with training modes, and applies one-hot encoding before fitting logistic regression.

## 5. Run the R Workflow

From the repository root, build the R image:

```powershell
docker build -t titanic-r -f src/r/Dockerfile .
```

Wait for the build to finish successfully, then run:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path)\src\data,target=/app/data,readonly" titanic-r
```

The image uses R 4.4.2 and installs `readr` through `install_packages.R`. Logistic regression uses R's included `stats` package.

The workflow fills missing numerical values with training medians, fills missing embarkation ports with the training mode, converts categorical predictors to factors, and fits logistic regression using `glm()` with a binomial family and logit link.

## 6. Features and Target

Both implementations use these predictors:

- `Pclass`
- `Sex`
- `Age`
- `Fare`
- `Embarked`
- `FamilySize`

The new feature is calculated as:

```text
FamilySize = SibSp + Parch + 1
```

It includes the passenger and their accompanying siblings, spouse, parents, and children.

The target is `Survived`:

- `0` means the passenger did not survive.
- `1` means the passenger survived.

Predicted survival probabilities are classified using a threshold of 0.5.

## 7. Expected Terminal Output

Both workflows print:

1. Confirmation that the training data loaded successfully.
2. Data structure, missing-value counts, and summary statistics.
3. Survival rates grouped by sex and passenger class.
4. Feature creation and preprocessing details.
5. Model training results.
6. Training accuracy as a percentage.
7. Predictions for the first 10 test passengers.
8. Counts of predicted survival outcomes.
9. An explanation of why test accuracy is unavailable.

The R workflow also prints a model summary and the total number of test predictions, which should be 418.

Training accuracy measures performance on the same passengers used to fit the model. Performance on unseen passengers requires a separate evaluation.

The provided `test.csv` has no true `Survived` labels, so local test accuracy cannot be calculated.

## 8. How the Data Mount Works

The run commands mount the computer's `src/data` directory at `/app/data` inside each container.

Both scripts read the CSV files from `/app/data`. The `readonly` option gives the containers read-only access to the source data.

The `--rm` option removes each container after its script finishes. The Docker images remain available for later runs.

## 9. Rerunning and Troubleshooting

- After editing a script, rebuild its image and run the corresponding container again.
- Run commands from the repository root.
- If a CSV file cannot be found, check that both files are directly inside `src/data` and that the full `--mount` argument is included.
- If Docker cannot connect to its engine, open Docker Desktop and wait for startup to complete.