conda create -y -n npl -c conda-forge -c pytorch --override-channels python=3.9 "numpy>=1.16.1" "pandas>=1.1.0" "geopandas>=0.10.2" psutil pyarrow numba pyyaml requests pytorch tensorflow spacy nltk wordcloud pytest pytest-cov coveralls pycodestyle pytest-regressions jupyter jupyterlab matplotlib descartes pandasql scipy seaborn pyodbc sqlalchemy openpyxl xlrd xlsxwriter sympy nose scikit-learn scikit-learn-intelex autopep8 black yapf pip ipykernel
conda activate npl