#!/bin/bash

if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Please install it using: pip install streamlit"
    exit 1
fi

streamlit run stats.py