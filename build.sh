#!/bin/bash
# MiniML build script
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"

echo "Building MiniML..."
g++ -O2 -std=c++17 -D_WIN32_WINNT=0x0A00 main.cpp -o miniml_new.exe -lws2_32 -lwsock32

if [ $? -eq 0 ]; then
    cp miniml_new.exe miniml.exe 2>/dev/null || mv miniml_new.exe miniml.exe
    echo "Build OK → miniml.exe"
else
    echo "Build FAILED"
    exit 1
fi
