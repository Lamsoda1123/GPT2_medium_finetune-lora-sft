@echo off
:: 设置环境变量
set model_name=你的model_name路径
set data_path=你的data_path路径
set lora=你的lora路径
set bs=20
set top_k=4
set top_p=0.3
set num_beams=5
set temperature=0.3

:: 使用特定的 Python 内核执行脚本
:loop
C:\Anaconda3\envs\myenv\python.exe %PYTHON_SCRIPT%
if %errorlevel% == 0 (
    echo Python 脚本成功执行。
    goto :eof
) else (
    echo Python 脚本执行失败，正在重试...
    timeout /t 1 /nobreak
    goto loop
)