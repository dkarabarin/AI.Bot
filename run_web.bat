@echo off
echo 🔥 Запуск веб-сервера термодинамического бота
echo ==============================================

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден. Установите Python 3.10+
    pause
    exit /b 1
)

REM Запуск сервера
echo.
echo 🚀 Запуск FastAPI сервера на http://localhost:8000
echo Нажмите Ctrl+C для остановки
echo.

python -m uvicorn web.api:app --reload --host 0.0.0.0 --port 8000

pause