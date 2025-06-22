@echo off
echo Starting HuggingFace NLP Playground...

REM Activate the virtual environment if it exists
if exist "hf-env\Scripts\activate.bat" (
    call hf-env\Scripts\activate.bat
) else (
    echo Virtual environment not found.
    echo Please ensure you've installed the requirements: pip install -r requirements.txt
    exit /b 1
)

REM Run the Streamlit app
echo Starting Streamlit application...
streamlit run src/app.py

REM Deactivate the virtual environment when done
call hf-env\Scripts\deactivate.bat
