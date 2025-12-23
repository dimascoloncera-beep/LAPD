@echo off
cd /d "%~dp0"

echo ==============================
echo  Iniciando Streamlit App
echo ==============================

if not exist "venv\Scripts\python.exe" (
  echo ERROR: No existe el entorno virtual en "venv".
  echo Crea el entorno con:  py -m venv venv
  pause
  exit /b 1
)

venv\Scripts\python -m streamlit run app.py

pause
