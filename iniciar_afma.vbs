Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\marco\Desktop\repo_asesor_git"
WshShell.Run "python -m streamlit run dashboard.py --server.headless true", 0, False
WScript.Sleep 5000
WshShell.Run "firefox http://localhost:8501", 1, False