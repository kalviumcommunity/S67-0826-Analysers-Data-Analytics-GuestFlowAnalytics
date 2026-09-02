import tempfile
import unittest
from pathlib import Path

import pandas as pd

from modules.data_processing import REQUIRED_COLUMNS, load_and_clean_data


class DataProcessingTests(unittest.TestCase):
    def make_row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {column: "" for column in REQUIRED_COLUMNS}
        row.update(
            {
                "hotel_name": "Test Hotel",
                "city": "Test City",
                "segment": "Direct",
                "total_rooms": 10,
                "check_in_date": "2025-01-15",
                "lead_time_days": 3,
                "length_of_stay": 2,
                "num_guests": 2,
                "num_rooms": 1,
                "adr": 100,
                "total_price": 200,
                "booking_status": "Completed",
                "is_cancelled": 0,
            }
        )
        row.update(overrides)
        return row

    def write_csv(self, rows):
        temporary_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        temporary_file.close()
        path = Path(temporary_file.name)
        pd.DataFrame(rows).to_csv(path, index=False)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_loads_valid_five_row_dataset(self):
        path = self.write_csv([self.make_row(check_in_date=f"2025-01-{day:02d}") for day in range(15, 20)])

        result = load_and_clean_data(path)

        self.assertEqual(len(result), 5)
        self.assertEqual(result.iloc[0]["month_name"], "January")

    def test_rejects_missing_required_columns(self):
        row = self.make_row()
        row.pop("city")
        path = self.write_csv([row])

        with self.assertRaisesRegex(ValueError, "missing required columns: city"):
            load_and_clean_data(path)

    def test_rejects_dataset_with_only_invalid_dates(self):
        path = self.write_csv([self.make_row(check_in_date="not-a-date")])

        with self.assertRaisesRegex(ValueError, "no valid check-in dates"):
            load_and_clean_data(path)

    def test_removes_duplicate_rows(self):
        row = self.make_row()
        path = self.write_csv([row, row])

        result = load_and_clean_data(path)

        self.assertEqual(len(result), 1)

    def test_applies_numeric_defaults(self):
        path = self.write_csv(
            [
                self.make_row(
                    total_rooms="",
                    lead_time_days="",
                    length_of_stay="",
                    num_guests="",
                    num_rooms="",
                    adr="",
                    total_price="",
                    is_cancelled="",
                )
            ]
        )

        result = load_and_clean_data(path).iloc[0]

        self.assertEqual(result["room_inventory"], 1)
        self.assertEqual(result["lead_time"], 0)
        self.assertEqual(result["total_nights"], 1)
        self.assertEqual(result["guest_count"], 1)
        self.assertEqual(result["rooms_booked"], 1)
        self.assertEqual(result["adr"], 0)
        self.assertEqual(result["revenue"], 0)


if __name__ == "__main__":
    unittest.main()