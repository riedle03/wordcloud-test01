AttributeError: 'FontManager' object has no attribute 'clear'
Traceback:
File "/home/vscode/.local/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 534, in _run_script
    exec(code, module.__dict__)
File "/workspaces/wordcloud-test01/streamlit_app.py", line 39, in <module>
    setup_korean_font()
File "/workspaces/wordcloud-test01/streamlit_app.py", line 23, in setup_korean_font
    fm.fontManager.clear()
    ^^^^^^^^^^^^^^^^^^^^