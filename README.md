# Multi-Disease Risk Prediction System (MRP)

A machine learning-based medical risk prediction system that helps healthcare providers assess patient risks for multiple disease categories including cardiovascular, metabolic, respiratory, and infectious diseases.

## Features

- Real-time risk prediction for multiple disease categories
- Interactive web interface built with Streamlit
- Advanced clinical measurement validation
- Personalized health recommendations
- Risk score calculation and trending
- Data visualization with interactive gauges and charts

## Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: scikit-learn (Random Forest Classifiers)
- **Database**: MySQL
- **Data Processing**: pandas, numpy
- **Visualization**: plotly

## Project Structure

- `schema/`: SQL migration files for database setup
- `src/`
  - `pipeline.py`: Data processing pipeline
  - `app/`: Main application directory
    - `model.py`: Machine learning models
    - `utils.py`: Utility functions for clinical validation and recommendations
    - `streamlit_app.py`: Web interface

## Getting Started

1. Set up the database:
```bash
python run_migrations.py
```

2. Install requirements:
```bash
cd src/app
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/app/streamlit_app.py
```

## Features in Detail

### Disease Risk Prediction
- Cardiovascular disease risk assessment
- Metabolic disorder risk calculation
- Respiratory disease risk evaluation
- Infectious disease risk prediction

### Clinical Measurements
- Blood pressure monitoring
- Heart rate tracking
- Blood glucose levels
- Cholesterol measurements
- Oxygen saturation
- Body Mass Index (BMI)
- White Blood Cell count
- C-Reactive Protein levels

### Health Analysis
- Risk score calculation
- Measurement trend analysis
- Clinical validation of measurements
- Personalized health recommendations
