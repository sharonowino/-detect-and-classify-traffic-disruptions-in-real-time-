# File: README.md

# GTFS Disruption Detection Pipeline

This project provides a multilingual real-time pipeline to analyze GTFS-Realtime service alerts, extract and classify disruption events using NLP and ML models.

## Features
- Multilingual NER, sentiment, and topic modeling
- Fusion of alert data with vehicle telemetry (speed, delay)
- Classification of disruption severity and type
- Real-time dashboard visualization using Streamlit

## Structure
- `config/`: Feed configuration
- `nlp/`: Language processing modules
- `mobility/`: Vehicle & alert data processing
- `fusion/`: Matching alerts to vehicle data
- `model/`: ML feature engineering and training
- `dashboard/`: Visualization with Streamlit

## How to Run
Install requirements and execute the dashboard:
```bash
pip install -r requirements.txt
streamlit run project/dashboard/streamlit_app.py
```
