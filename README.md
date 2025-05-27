
python 3.13.0


```
# 依存インストール
pip install pybind11
brew install opencv cmake

# ビルド
(.venv) ebiharamari@Ebiharas-MacBook-Pro spherical_warper_pybind % cd build
rm -rf *
cmake .. -DPYTHON_EXECUTABLE=../.venv/bin/python
make

# Pythonで実行
cd ..
python3 main.py
```
