@echo off
cd /d "%~dp0"
C:\Python314\python.exe -B -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost --server.headless true
