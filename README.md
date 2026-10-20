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
