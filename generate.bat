@echo off
:: ���û�������
set model_name=���model_name·��
set data_path=���data_path·��
set lora=���lora·��
set bs=20
set top_k=4
set top_p=0.3
set num_beams=5
set temperature=0.3

:: ʹ���ض��� Python �ں�ִ�нű�
:loop
C:\Anaconda3\envs\myenv\python.exe %PYTHON_SCRIPT%
if %errorlevel% == 0 (
    echo Python �ű��ɹ�ִ�С�
    goto :eof
) else (
    echo Python �ű�ִ��ʧ�ܣ���������...
    timeout /t 1 /nobreak
    goto loop
)