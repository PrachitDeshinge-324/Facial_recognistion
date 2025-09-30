#!/bin/bash

# clean.sh - Clear all cache files and temporary files from the face recognition project

echo "🧹 Cleaning Face Recognition Project Cache..."

# Remove Python cache files
echo "Removing Python __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null

# Remove macOS system files
echo "Removing macOS system files..."
find . -name ".DS_Store" -delete 2>/dev/null

# Remove IDE cache files
echo "Removing IDE cache files..."
rm -rf .vscode/settings.json 2>/dev/null
rm -rf .idea/ 2>/dev/null
rm -rf *.code-workspace 2>/dev/null

# Remove Python virtual environment cache
echo "Removing Python virtual environment cache..."
rm -rf .venv/ 2>/dev/null
rm -rf venv/ 2>/dev/null
rm -rf env/ 2>/dev/null

# Remove Jupyter notebook checkpoints
echo "Removing Jupyter notebook checkpoints..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# Remove temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -delete 2>/dev/null
find . -name "*.temp" -delete 2>/dev/null
find . -name "*~" -delete 2>/dev/null

# Remove log files (optional - comment out if you want to keep logs)
echo "Removing log files..."
find . -name "*.log" -delete 2>/dev/null

# # Remove output directory contents (but keep the directory)
# if [ -d "output" ]; then
#     echo "Cleaning output directory..."
#     rm -rf output/* 2>/dev/null
# fi

# Remove build artifacts
echo "Removing build artifacts..."
rm -rf build/ 2>/dev/null
rm -rf dist/ 2>/dev/null
rm -rf *.egg-info/ 2>/dev/null

# Remove coverage files
echo "Removing coverage files..."
rm -rf .coverage 2>/dev/null
rm -rf htmlcov/ 2>/dev/null
rm -rf .pytest_cache/ 2>/dev/null

# Remove any downloaded model cache (InsightFace models)
# echo "Removing InsightFace model cache..."
# rm -rf ~/.insightface/ 2>/dev/null

echo "✨ Cache cleanup completed!"
echo ""
echo "📁 Preserved directories:"
echo "  - face_database/ (your source images)"
echo "  - database/ (your generated database files)"
echo "  - config/ (your configuration files)"
echo "  - src/ (your source code)"
echo ""
echo "🗑️  Cleaned:"
echo "  - Python cache (__pycache__/)"
echo "  - macOS system files (.DS_Store)"
echo "  - IDE cache files"
echo "  - Temporary files"
echo "  - Log files"
echo "  - Build artifacts"
echo "  - Virtual environment cache"
echo "  - InsightFace model cache"