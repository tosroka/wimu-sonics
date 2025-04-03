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

# Wnioski z paperów:
1. "SONICS: Synthetic Or Not -- Identifying Counterfeit Songs"
* Poprzednie modele które przytaczają (Singing Voice Synthesis i Singing Voice Conversion) korzystały z generowanego głosu ale instrumenty pochodziły z prawdziwych piosenek. Powodowało to "Karaoke effect" rozbieżności w głośności. Poza tym nie zwracały one uwagi na długoterminowe relacje czasowe takie jak refren itp.
* Przygotowali swój własny dataset, średnii  czas utworu to 3 min żeby badać dłuższy kontekst
* Wykorzystują Spectro-Temporal Tokens Transformer (SpecTTTra) jako architekturę, korzysta z Spectro-Temporal Tokenizer
*  prawdziwe piosenki brali z Genius Lyrics Datase, nie jest z całego więc można zautomatyzować pobranie rzeczywistych jak byśmy chcieli więcej 
* Używali do generowania Suno v2, v3, and v3.5. Suno v2, i Udio 32 and Udio 130, 
* Do generowanie stosowane słowa piosenki oraz styl  (lyrics, style). Podział fakowych na: Full Fake (FF) — teskt i styl generowany losowo.
Mostly Fake (MF) - tekst generowany podobny do podanego prawdziwego tekstu a styl brany na podstawie prawdziwej piosenki; Half Fake (HF)—
tekst brany prawdziwy i styl też pobierany prawdziwy [Udio zabrania czegoś takiego na poziomie API]
* Do modelu używali Mel-spectogramu, biorą go oraz jego transpozcyja i dzielą go na fragmenty żeby odpowiednio analizować zmiany w czasie(temporal) oraz zakresy częstotliowości (spectral). Podawane są one potem oddzielnie do odpowiednich tokenizerów i na koniec tokeny i embedingi są przekazywane do transformera który przypomina VIT
* w treningu nie używali wszystkich modeli (Suno v2, Suno v3, Udio 32) oraz nie wszystkich twórców aby zapewnić i też sprawdzić że model uczy się ogólnej cechy czy muzyka jest generowana
* preprocessing: 16kHz, spectogramy: n_fft=win_length=2048, hop_lenght=512, n_mels=128. Yielding a 128 × 128 spectrogram
for 5 sec and 128 × 3744 for 120 sec audio. Any song shorter than input length is zero-padded
randomly, while for longer songs, a random crop is use
* Korzystają z augmentacji z  MixUp oraz SpecAugment TODO: sprawdzić jakeigo typu to augmentacje i najlepiej znaleźć jakąś której nie przewidzieli
* Mają wniosek że łatwiej wykryć że utwór jest prawdziwy niż fałszywy, ten sam wniosek co w kolejnym artykule a w sumie go nie cytowali
* Modele generatywne na których nie było trenowane działają gorzej ale nie jakoś bardzo, trzeba tylko zauważyć że biorą oni modele z tej samej rodziny tylko starsze wersje więc mają one pewnie jakieś wspólne cechy do tych nowszych
* Jako ciekawostka raczej że stworzyli human-benchmark i daje on gorsze wyniki niż ich modelu
* w apendix są statystki ich datasetu i składa się głównie z rock, folk, pop
* zrobili też mniejsze testy na SkyMusic and SeedMusic, wyniki które pokazali znacząco spadły do 60% na Sky music dla najmniejszego modelu, a większe poniżej 30% ze względu na overfitting. Ale i tak pokazują że ich architektura działa lepiej niż dotychczasowe
* jeżeli chcielbyśmy coś generować jak tutaj w paperze to są podane prompty w paperze
### Charakterystyka piosenek względem przypisanych klas:
* In True Negative cases, we find distinct patterns
in correctly classified real songs. These include characteristics such as unpredictability, dynamic
variation, and unexpected changes that is often absent in fake songs. Examples include non-standard
pitch variations, intricate rhythmic complexity, and expressive techniques like melismatic phrasing,
sudden tempo changes, or improvisational segments, all of which showcase the nuanced artistry
of human performance
* in True Positive cases, we detect specific audible artifacts in correctly classified fake songs. These artifacts include mechanical or robotic vocal qualities, unclear
vocal articulation, predictable rhythmic structures, and limited pitch variability. Such fake songs
also lack the emotive expressiveness and complexity we consistently find in real music, making
them notably distinct.
* In False Negative cases, we observe that fake songs not detected as such
lack the typical artifacts seen in true positive fake songs. Instead, these cases often incorporate
features that mimic the unpredictability and nuanced variation of real songs. For instance, some
include spoken interludes or conversational segments before the singing starts, creating a deceptive
resemblance to genuine music.
* In False Positive cases, we find that real songs misclassified as
fake share characteristics with AI-generated music. These include unclear vocals, less rhythmic
complexity, and a noticeable absence of the unexpected changes that typically distinguishes human-
created performances. 
* Finally, we encounter instances where we are unable to detect any audible artifacts. This suggests the presence of subtle, imperceptible, or inaudible artifacts akin to invisible artifacts in synthetic images.
  
2. "DETECTING MUSIC DEEPFAKES IS EASY BUT ACTUALLY HARD"
* dostępne repozytorium z kodem: https://github.com/deezer/deepfake-detector, jest dostępny jakiś lepszy model ale im się nie dzielą na razie?
* na konferencji IEEE ICASSP 2025, 6-11 kwietnia może coś więcej będzie
* skupia się na "waveform generators", a nie symbolicznych w papierze jest podane więcej generatorów niż te co używamy ale chyba te co mamy nam wystarczą
* prosta konwolucja dała im wyniki ok. 90%, ale w dalszej części papera opisują że następujące aspekty nie są rozpatrywane: robustness to manipulation, generalisation to unseen generators,
calibration and interpretability
* modele do detekcji mogą uczyć się charakterystyk konkretnych modeli generatywnych: MusicGen might always interpret a text prompt in a
certain way and in 4/4. Inny przyklad to w dekoderach checkerboard artefacts dla operacji dekonwolucji. Mogą też dopasowywać się do zbiorów danych, na których były uczony detektor np. publiczny zbiór danych dotyczących muzyki może być pełen muzyki klasycznej, podczas gdy zbiór danych deepfake obejmuje głównie muzykę rap i pop. Wtedy model będzie wykrywał muzykę klasyczną zamiast deepfaki. Ten sam problem może pojawiać się gdy compression codec that might confound the detection of deepfakes (e. g., all Riffusion songs are exported in mp3 192kB/s).
* trenowali na FMA dataset, zachowują ten sam bitrate, ale normalizują do 44.1kHz
* Testowali 3 autoencodery: Encodec, DAC oraz własny GriffinMel będący połączeniem mel-spektogramu i fazy Griffina-Lima. Testowali ich różne kombinacje dostępne na https://research.deezer.com/deepfake-detector/. Badali na reprezentacjach: the raw waveform, the complex STFT, its amplitude, its phase, or both stacked as polar coordinates. Stosowali do pre-processingu  normalisation, random mono mix, cutoff at 16kHz,
and conversion to decibel scale when applicable.
* Testowali modyfikacje: random pitch shift
(±2 semitones), time stretch ([80, 120]%), EQ, reverb, addition of white noise, reencoding in mp3, aac, and opus in
64kB/s. Implementation details are available in our repository.We leave attacks from more advanced users for future
work (e. g., adversarial attacks). Po większości tych modyfikacji wynik modelu drastycznie spada, sprawdzali te modyfikacje też na prawdziwych danych i dalej zwracały te same wyniki czego wnioskiem jest że model sprawdza czy nie ma wyuczonych cech fałszywych. Nie trenowali modelu ponownie po tych zmianach ponieważ: 'there will always be an unseen manipulation'
* Encoder generalisation: ponownie trenowali najlepszy model od podstaw na każdym z dziewięciu rozważanych dekoderów i sprawdzali, jak wydajność wykrywania przenosi się na pozostałe. I znowu można zauważyć spadek nawet do 0% w wykrywaniu bo klasa że muzyka jest prawdziwa jest domyślnie przewidywana
* Sprawdzanie jak wykrywa gdy np. śpiew generowany a instrumenty prawdziwe, ale sami nie wiedzą co to powinno zwracać tylko zwraca to uwagę
na to że należy ustalać specyfikację co powinien wykrywać
* Interpretacja: Podmieniali losowo częsci spektrogramów prawdziwych fałszywmi fragmentami i pokazali że wykrywane są one na feature attribution maps. Piszą też że można wykorzystać concept learning, jeżeli cechą może być: 'spectrogram was related to its overall bluriness'
3. "AI-Generated Music Detection and its Challenges"
* praktycznie to samo co poprzedni paper tylko przygotowany na konferencję
4. "YuE: Scaling Open Foundation Models for Long-Form Music Generation"
5. "Simple and Controllable Music Generation"
6.  "Transformer Interpretability Beyond Attention Visualization"
7.  Może uda się zaadaptować do tego problemu SHAP https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/question_answering/Explaining%20a%20Question%20Answering%20Transformers%20Model.html
8. "Fake speech detection using VGGish with attention block"

## Dodatkowy research potrzebny:
* mixup, https://arxiv.org/pdf/1710.09412
* SpecAugment, https://arxiv.org/pdf/1904.08779

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
