#!/bin/bash
# Script premium para compilar el paper de Ontogenia Artificial

# Detener en caso de error
set -e

echo "============================================="
echo "🚀 INICIANDO COMPILACIÓN DEL PAPER (LaTeX) 🚀"
echo "============================================="

# 1. Compilación inicial
echo -e "\n📝 [1/4] Ejecutando pdflatex (primera pasada)..."
pdflatex -interaction=nonstopmode main.tex

# 2. Compilar bibliografía
echo -e "\n📚 [2/4] Procesando citas bibliográficas con bibtex..."
bibtex main

# 3. Segunda compilación para resolver referencias
echo -e "\n🔄 [3/4] Ejecutando pdflatex (segunda pasada)..."
pdflatex -interaction=nonstopmode main.tex

# 4. Tercera compilación para fijar numeración de figuras y páginas
echo -e "\n🎯 [4/4] Ejecutando pdflatex (tercera pasada final)..."
pdflatex -interaction=nonstopmode main.tex

echo -e "\n============================================="
echo "🎉 ¡PAPER COMPILADO CON ÉXITO! 🎉"
echo "📄 Archivo generado: paper/main.pdf"
echo "============================================="
