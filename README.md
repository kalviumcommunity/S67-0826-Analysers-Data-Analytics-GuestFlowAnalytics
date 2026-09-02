# GuestFlow Analytics

GuestFlow Analytics is a Streamlit application for analyzing hotel booking data. It prepares booking data, calculates hotel performance metrics, and stores the cleaned records in SQLite for analytics queries.

## Requirements

- Python 3.10 or newer
- A CSV dataset with the required booking columns

## Setup

From the project directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Dataset

Place the input file at:

```text
data/guestflow_cleaned_data.csv
```

The CSV must include these columns:

```text
segment
hotel_name
city
total_rooms
check_in_date
lead_time_days
length_of_stay
num_guests
num_rooms
adr
total_price
booking_status
is_cancelled
```

A small five-row CSV can be used for local development and testing. Duplicate rows are removed, numeric missing values receive defaults, and rows without valid `check_in_date` values are rejected.

## Run the application

Start the Streamlit application with:

```powershell
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a browser.

## Database

When data is prepared, the application creates the SQLite database automatically at:

```text
database/hotel_data.db
```

The cleaned booking records are stored in the `bookings` table.

## Tests

Run the data-processing tests with Python's built-in test runner:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
