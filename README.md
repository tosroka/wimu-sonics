# wimu-sonics

## Opis tematu
Projekt ma na celu zweryfikować pracę "SONICS: Synthetic Or Not -- Identifying Counterfeit Songs" pod kątem wykrywania wygenerowanej muzyki przez otwarte modele (YuE, MusicGen), oraz sprawność na oryginalnym zbiorze danych z różnymi modyfikacjami. Następnie spróbujemy przeprowadzić wyjaśnialność modelu za pomocą narzędzi [Transformer Explainability](https://github.com/hila-chefer/Transformer-Explainability), SHAP lub innych, o ile uda się je zaadaptować. Chcemy zweryfikować, czy model faktycznie bierze pod uwagę długodystansowe korelacje w utworze, tak jak napisane jest w paperze, czy polega na artefaktach w muzyce z Suno i Udio.


## Planowany rozkład jazdy

#. 17.03-23.03  
  * Zapoznanie się z modelami SONICS do klasyfikacji oraz YuE i MusicGen do generacji. Pobranie ich lokalnie i wstępne uruchomienie
  * Pobranie opublikowanego zbioru Suno i Udio
  * Stworzenie szablonu cookiecutter
  * (?) Dostęp do maszyny do przeprowadzania eksperymentów

#. 24.03-30.03
  * Utworzenie mini-zbiorów danych do wstępnego zweryfikowania modeli: po 10 losowych utworów z Suno, Udio, MusicGen, YuE oraz prawdziwych np. z YouTube
 
#. 31.03-06.04
  * Napisanie szablonów oraz kodu do wygenerowania tekstu piosenek do YuE, oraz promptów do MusicGen dla większego datasetu (na wzór tego, co zrobili w pracy SONICS)

#. 07.04-13.04 (PROTOTYP - deadline)
  * ciąg dalszy poprzedniego
  * Spisanie wniosków z paperów

#. 14.04-20.04
  * Napisanie kodu do modyfikacji zbiorów danych - pitch shift, zmiana sample rate, pogłos i inne

#. 21.04-27.04
  * Napisanie struktury kodu do powtarzalnych eksperymentów. Chcemy jedną komendą uruchamiać wszystkie badania i otrzymywać wyniki

#. 28.04-04.05
  * ciąg dalszy poprzedinego / majówka
  

#. 05.05-11.05
  * Kod do wizualizacji wyników
  * uruchomienie narzędzi do wyjaśnialności

#. 12.05-18.05
  * Draft na LBD ISMIR2025
  * Wstawić nasz wygenerowany zbiór danych na zenodo

#. 19.05-25.05
  * Stworzenie prezentacji

#. 26.05-01.06
  * Tydzień na potencjalne opóźnienia


## Bibliografia

1. "SONICS: Synthetic Or Not -- Identifying Counterfeit Songs" https://arxiv.org/abs/2408.14080
  * Główny paper
2. "DETECTING MUSIC DEEPFAKES IS EASY BUT ACTUALLY HARD" https://arxiv.org/pdf/2405.04181
3. "AI-Generated Music Detection and its Challenges"  https://arxiv.org/abs/2501.10111
  * drobne rozwinięcie poprzedniego papera, przedstawia różne techniki augumentacji danych przydatne do weryfikowania modeli klasyfikujących sztuczne audio. Podejrzewamy podobne wyniki
4. "YuE: Scaling Open Foundation Models for Long-Form Music Generation" https://arxiv.org/abs/2503.08638
5. "Simple and Controllable Music Generation" - https://arxiv.org/abs/2306.05284
6.  "Transformer Interpretability Beyond Attention Visualization" https://arxiv.org/abs/2012.09838
  * Interpretowalność transformerów na poziomie wchodzących tokenów
7.  Może uda się zaadaptować do tego problemu SHAP https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/question_answering/Explaining%20a%20Question%20Answering%20Transformers%20Model.html
8. "Fake speech detection using VGGish with attention block" https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00348-4
  * Trochę ciekawych referencji oraz sposoby augumentacji danych w ramach uczenia modeli do detekcji fake audio

## Zakres eksperymentów

## Stack technologiczny (biblioteki python)
* Librosa/torchaudio - transformacje audio
* huggingface - modele SONICS i MusicGen
* openai - generowanie słów do utworów na podstawie szablonów

Project Organization
------------

    ├── LICENSE             <- MIT license file
    ├── Makefile            <- Makefile with commands like `make data` or `make train`
    ├── README.md           <- The top-level README for individuals using this project
    ├── data
    │   ├── intermediate    <- Intermediate data that has been transformed
    │   ├── processed       <- The final, canonical data sets for modeling
    │   └── raw             <- The original, immutable data dump
    │
    ├── docs                <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models              <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                          the creator's initials, and a short `-` delimited description, e.g.
    │                          `1.0-ilm-initial-data-exploration`.
    │
    ├── references          <- Data dictionaries, manuals, and all other explanatory materials
    │
    ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures         <- Generated graphics and figures to be used in reporting
    |
    ├── results             <- Generated results from data analysis and fitting models
    │
    ├── src                 <- Source code for use in this project
    │   ├── __init__.py     <- Makes src a Python module
    │   │
    │   ├── data            <- Scripts to load and process data
    │   │   └── load_data.py
    |   |   └── create_int_data
    │   │   └── create_pro_data.py
    │   │
    │   ├── models          <- Scripts for models and fitting processed data
    │   │   └── model.py
    │   │
    │   └── visualization   <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                          generated with `pip freeze > requirements.txt`
    │
    ├── setup.py            <- makes project pip installable (pip install -e .) so src can be imported
    |
    └── test_environment.py <- checks that correct python interpreter is installed


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
