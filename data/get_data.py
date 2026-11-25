import yfinance as yf
import pandas as pd

def get_stock_prices(tickers, begin_date, end_date, filename="stock_prices.csv"):
    """
    Downloads daily closing prices for multiple tickers
    and saves them into a CSV file.

    Parameters:
        tickers: List of ticker symbols (e.g., ['GOOG', 'JPM', 'AMZN'])
        begin_date: Starting date ('YYYY-MM-DD')
        end_date: Ending date ('YYYY-MM-DD')
        filename: Output CSV filename
    """

    # Download historical data
    data = yf.download(
        tickers,
        start=begin_date,
        end=end_date,
        auto_adjust=False
    )

    # Extract closing prices
    close_prices = data['Close']

    # Save to CSV
    close_prices.to_csv(filename)

    return close_prices


# Date range
begin_date = "2025-01-01"
end_date = "2026-01-02"

# Tickers
tickers = ["GOOG", "JPM", "AMZN"]

# Download and save
prices = get_stock_prices(tickers, begin_date, end_date)

print(prices.head())