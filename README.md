# Fake News Detection Project

## Overview
This project is a Fake News Detection application built using Streamlit and TensorFlow. It leverages a pre-trained BERT model to classify news articles as real or fake. The application fetches real-life news from various sources, processes the data, and provides users with an intuitive interface to analyze the credibility of news headlines.

## Project Structure
```
fake-news-streamlit
├── app.py                   # Main entry point of the Streamlit application
├── src                      # Source code for the application
│   ├── news_fetcher.py      # Functions to fetch real-life news
│   ├── model_loader.py      # Loads the pre-trained machine learning model
│   ├── predictor.py         # Prediction logic for classifying news
│   ├── preprocess.py        # Preprocessing functions for fetched news data
│   ├── visualizer.py        # Visualization of prediction results
│   └── utils.py             # Utility functions used across the application
├── models                   # Directory for model documentation
│   └── README.md            # Documentation about the models used
├── notebooks                # Jupyter notebooks for exploration
│   └── exploration.ipynb    # Notebook for exploratory data analysis
├── tests                    # Unit tests for the application
│   └── test_predict.py      # Tests for the prediction logic
├── requirements.txt         # Project dependencies
├── Dockerfile               # Instructions for building a Docker image
├── .env.example             # Example environment variables
├── .gitignore               # Files and directories to ignore by Git
└── README.md                # Project documentation
```

## Setup Instructions
1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd fake-news-streamlit
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Copy `.env.example` to `.env` and fill in the required values.

5. **Run the application**:
   ```
   streamlit run app.py
   ```

## Usage
- The application provides a dashboard for users to input news headlines and receive predictions on their credibility.
- Users can also upload CSV files for batch predictions.
- The sidebar contains controls for adjusting model parameters and viewing token importance.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
