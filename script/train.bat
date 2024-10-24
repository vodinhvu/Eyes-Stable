@echo off

REM Load configuration from the config file
for /f "tokens=1,* delims==" %%A in (config\config.txt) do (
    set %%A=%%B
)

REM Download the dataset
echo Downloading dataset from %DATASET_URL%...
set DATASET_DIR=data

REM Create data directory if it doesn't exist
if not exist "%DATASET_DIR%" (
    mkdir "%DATASET_DIR%"
)

REM Download the dataset using PowerShell
powershell -Command "Invoke-WebRequest -Uri '%DATASET_URL%' -OutFile '%DATASET_DIR%\dataset.zip'"

REM Unzip the dataset
powershell -Command "Expand-Archive -Path '%DATASET_DIR%\dataset.zip' -DestinationPath '%DATASET_DIR%'"

REM Navigate to the src directory
cd src

REM Run main.py with arguments loaded from the config file
echo Running main.py...
python main.py --image "%IMAGE_DIR%" --epochs %EPOCHS% --save "..\%SAVE_DIR%" --batch %BATCH% --num_workers %NUM_WORKERS%
