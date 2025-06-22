@echo off
echo Installing requirements for HuggingFace NLP Playground...

REM Activate the virtual environment
call hf-env\Scripts\activate.bat

REM Install the requirements
pip install -r requirements.txt

echo Installation completed!
echo Run 'run_app.bat' to start the application.

REM Deactivate the virtual environment
call hf-env\Scripts\deactivate.bat
