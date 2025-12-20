@echo off
echo Compiling Neural Voice Conversion Technical Report...
echo.

pdflatex technical_report.tex
pdflatex technical_report.tex

echo.
echo Compilation complete! Check technical_report.pdf
pause
