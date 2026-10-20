# ChatBot TODO + NLP (Slack)

Bot w Slacku do zarządzania listą zadań TODO, który wykonuje operacje NLP na treści zadań.

## Pliki

- `chatbot.ipynb` – cały kod bota: baza zadań (SQLite), operacje NLP, komendy i połączenie ze Slackiem
- `slack_manifest.yml` – konfiguracja aplikacji Slack

## Uruchomienie

1. Na [api.slack.com/apps](https://api.slack.com/apps) kliknij **Create New App → From a manifest**
   i wklej zawartość `slack_manifest.yml`.
2. **Basic Information → App-Level Tokens**: wygeneruj token ze scope `connections:write` (`xapp-...`).
3. **Install App → Install to Workspace** i skopiuj **Bot User OAuth Token** (`xoxb-...`).
4. Zainstaluj biblioteki (wymagany Python 3.12) i otwórz notatnik:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # wpisz tokeny i numer albumu
jupyter notebook chatbot.ipynb
```

5. Uruchom komórki po kolei. Ostatnia sekcja startuje bota, który działa, dopóki notatnik jest otwarty.

## Komendy

| Komenda | Opis |
| --- | --- |
| `/tasks` | wyświetla wszystkie zadania |
| `/add [treść]` | dodaje zadanie; `/add [treść] \| [id]` dodaje podzadanie zadania o podanym id |
| `/update [id] [nowa treść]` | zmienia treść zadania; `/update [id] status=done` zmienia status |
| `/delete [id]` | usuwa zadanie |
| `/task [id] [operacja]` | wykonuje operację NLP na treści zadania |
| `/prepare_data_for_classification [todo\|20newsgroups]` | nadaje etykiety zadaniom TODO (`praca`, `dom`, `nauka`) albo wczytuje 4 tematy z 20 Newsgroups; wybrany zbiór staje się aktywny |
| `/classifier cleaning "tekst" [bow\|tfidf\|embedding] [svm\|logreg]` | klasyfikuje tekst (po `clean_text`) i zwraca klasę z prawdopodobieństwem |
| `/evaluate [bow\|tfidf\|embedding] [svm\|logreg]` | macierz pomyłek oraz ACC, ERR, Precision, Recall i F1 na zbiorze testowym |

Przykład:

```text
/add Przygotować prezentację z przetwarzania języka naturalnego
Utworzono zadanie: 550e8400-e29b-41d4-a716-446655440000

/task 550e8400-e29b-41d4-a716-446655440000 bigram
[('przygotować', 'prezentację'), ('prezentację', 'z'), ('z', 'przetwarzania'), ...]
```

## Struktura zadania

| Pole | Opis |
| --- | --- |
| `id` | UUID zadania |
| `description` | treść zadania |
| `relation` | id zadania nadrzędnego (opcjonalne) |
| `status` | `todo`, `in_progress` lub `done` |
| `created_at` | data utworzenia |
| `updated_at` | data ostatniej modyfikacji |
| `label` | klasa zadania nadawana przez `/prepare_data_for_classification` |

## Operacje NLP

| Operacja | Opis |
| --- | --- |
| `tokens` | tokenizacja NLTK i spaCy obok siebie (porównanie) |
| `clean` | `clean_text`: małe litery, bez liczb, interpunkcji, znaków specjalnych, pojedynczych znaków i nadmiarowych spacji |
| `normalize` | normalizacja Unicode (NFKC), ujednolicenie cudzysłowów i myślników, małe litery, białe znaki; dodatkowo wersja bez polskich znaków |
| `unigram`, `bigram`, `trigram` | n-gramy (NLTK) |
| `xgram` | indywidualny x-gram – x z ostatniej cyfry numeru albumu (4–9 → x, 0–3 → cyfra + 4) |
| `bow` | Bag of Words (`CountVectorizer`) |
| `tfidf` | TF-IDF (`TfidfVectorizer`) |
| `stem` | stemming – algorytm Stempel (NLTK nie ma stemmera dla polskiego) |
| `lemma` | lematyzacja spaCy (`pl_core_news_sm`) |
| `stopwords` | tekst oryginalny i tekst bez stopwords (polska lista ze spaCy, bo NLTK jej nie ma) |

## Parametry Bag of Words i TF-IDF

Słownik jest budowany na treściach wszystkich zadań, a bot pokazuje wektor wybranego zadania.

| Parametr | Wartość | Znaczenie |
| --- | --- | --- |
| `lowercase` | `True` | zamiana tekstu na małe litery przed liczeniem słów |
| `ngram_range` | `(1, 2)` | w słowniku są pojedyncze słowa i pary sąsiednich słów |
| `max_features` | `50` | słownik ma maksymalnie 50 najczęstszych terminów |
| `min_df` | `1` | termin musi wystąpić w co najmniej 1 zadaniu |
| `max_df` | `1.0` | nie odrzucamy terminów występujących we wszystkich zadaniach (przy małej liczbie zadań mniejsza wartość usunęłaby prawie wszystko) |
| `binary` (BoW) | `False` | liczymy, ile razy słowo wystąpiło; `True` dawałoby tylko 0 lub 1 |
| `use_idf` (TF-IDF) | `True` | słowa występujące w wielu zadaniach dostają mniejszą wagę |
| `smooth_idf` (TF-IDF) | `True` | do liczników IDF dodawane jest 1, żeby uniknąć dzielenia przez zero |
| `sublinear_tf` (TF-IDF) | `True` | zamiast liczby wystąpień używane jest 1 + log(TF), żeby powtórzenia nie dominowały |

## Klasyfikacja tekstu

- **Metody:** Linear SVM (`LinearSVC` + `CalibratedClassifierCV` dla prawdopodobieństw) i Logistic Regression – alfabetycznie najbliższe literze K.
- **Reprezentacje:** BoW, TF-IDF oraz embedding `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers).
- **Dane:** 75 zadań TODO z etykietami z słów kluczowych oraz 3807 wiadomości z 20 Newsgroups (grafika, hokej, medycyna, kosmos); podział 80/20, stratyfikowany, `random_state=42`.

| Zbiór | Reprezentacja | Metoda | ACC | ERR | F1 (macro) |
| --- | --- | --- | --- | --- | --- |
| 20 Newsgroups | BoW | SVM | 0.837 | 0.163 | 0.837 |
| 20 Newsgroups | BoW | LogReg | 0.854 | 0.146 | 0.855 |
| 20 Newsgroups | TF-IDF | SVM | 0.916 | 0.084 | 0.916 |
| 20 Newsgroups | TF-IDF | LogReg | 0.899 | 0.101 | 0.899 |
| 20 Newsgroups | Embedding | SVM | **0.927** | **0.073** | **0.926** |
| 20 Newsgroups | Embedding | LogReg | 0.923 | 0.077 | 0.922 |
| TODO | BoW | SVM | 0.600 | 0.400 | 0.589 |
| TODO | BoW | LogReg | 0.533 | 0.467 | 0.484 |
| TODO | TF-IDF | SVM | 0.667 | 0.333 | 0.667 |
| TODO | TF-IDF | LogReg | 0.733 | 0.267 | 0.731 |
| TODO | Embedding | SVM | 1.000 | 0.000 | 1.000 |
| TODO | Embedding | LogReg | 1.000 | 0.000 | 1.000 |

Macierze pomyłek, metryki dla każdej klasy, walidacja na własnych zdaniach i wnioski są w notatniku (sekcja 6).
