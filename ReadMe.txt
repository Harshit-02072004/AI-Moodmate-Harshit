# Sentiment-Based Song Recommender

## 📝 Overview

This project is a simple yet powerful song recommender that suggests music based on the sentiment of user-provided text. The user enters a sentence or a few words describing their mood, and the program analyzes the text to determine a sentiment score. Based on this score, it recommends a song from a vast dataset of over 170,000 tracks that matches the emotional tone.

The project is built within a Jupyter Notebook (`.ipynb`) for an interactive and easy-to-follow experience.

## ✨ Features

  - **Sentiment Analysis:** Analyzes text input to classify the mood into five distinct categories: *Very Sad, Sad, Neutral, Happy,* and *Very Happy*.
  - **Score-Based Matching:** Uses a sentiment score ranging from -1 (negative) to +1 (positive).
  - **Database-Driven Recommendations:** Recommends songs by intelligently filtering a large CSV database (`data.csv`).
  - **Musical Feature Mapping:** Maps the calculated sentiment score to the `valence` audio feature of songs, which represents musical positiveness (e.g., sad songs have low valence, happy songs have high valence).

## ⚙️ How It Works

1.  **User Input:** The program prompts the user to enter text describing their current mood.
2.  **Sentiment Calculation:** The input text is processed by **NLTK's VADER** (Valence Aware Dictionary and sEntiment Reasoner), which calculates a normalized `compound` sentiment score.
3.  **Mood Categorization:** The compound score is mapped to one of the five sentiment categories.
4.  **Database Filtering:** Each sentiment category corresponds to a specific range of the `valence` musical feature. The program filters the `data.csv` DataFrame to find all songs that fall within the target valence range.
5.  **Song Suggestion:** A random song is selected from the filtered list and presented to the user as a recommendation.

## 📦 Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

  - Python 3.x
  - Jupyter Notebook or Jupyter Lab

### 1\. Place Files

Ensure the following files are in the same directory:

  - The Jupyter Notebook file (`.ipynb`) containing the code.
  - The song database: `data.csv`.

### 2\. Install Required Libraries

Open your terminal or command prompt and run the following command to install the necessary Python libraries:

```bash
pip install pandas nltk
```

### 3\. Download NLTK VADER Lexicon

The first time you run the project, you'll need to download the VADER lexicon, which NLTK uses for its analysis. This is handled by a code cell within the notebook, which runs the following command:

```python
import nltk
nltk.download('vader_lexicon')
```

## 🚀 Usage

1.  **Start Jupyter:** Open your terminal, navigate to the project directory, and start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  **Open the Notebook:** Click on the `.ipynb` file from the Jupyter interface in your browser.
3.  **Run the Cells:** Execute the notebook cells in sequential order from top to bottom.
4.  **Enter Your Mood:** When you run the final cell, you will see an input prompt. Type a sentence describing how you feel and press `Enter`.
5.  **Get Your Recommendation:** The output will display the detected sentiment, the score, and a song title with its artist.

**Example Interaction:**

```
How are you feeling today? Describe it in a few words: > I had a terrible and awful day.

-------------------------------------
Sentiment Analysis: Very Sad 😭
Sentiment Score: -0.81
-------------------------------------
✨ Song Suggestion For You ✨
'Come as You Are' by Nirvana
-------------------------------------
```

## 🛠️ Technologies Used

  - **Python:** The core programming language.
  - **Jupyter Notebook:** For interactive development and presentation.
  - **Pandas:** For loading and manipulating the song database.
  - **NLTK (Natural Language Toolkit):** Used for its powerful, pre-trained VADER sentiment analysis tool.