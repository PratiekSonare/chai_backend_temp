"""
Tool functions for fetching and manipulating data
"""
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import json
import calendar

def _fetch_orders_window(start_date: str, end_date: str, api_key: str, jwt_token: str, base_url: str) -> List[Dict]:
    """Fetch all orders for a date window with pagination support"""
    all_orders = []
    url = f"{base_url}/orders/V2/getAllOrders"
    
    params = {
        "limit": 250,
        "start_date": start_date,
        "end_date": end_date
    }
    
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    
    page = 1
    
    while True:
        try:
            print(f"Fetching page {page} for date range {start_date} to {end_date}")
            response = requests.get(url, params=params, headers=headers)
            
            # Check if we got a 400 Bad Request (end of pagination)
            if response.status_code == 400:
                print(f"Reached end of pagination (400 Bad Request) at page {page}")
                break
                
            response.raise_for_status()
            
            data = response.json()
            if data.get("code") != 200 or "data" not in data:
                print(f"API returned non-200 code or missing data: {data}")
                break
            
            # Extract orders from current page
            page_orders = data["data"].get("orders", [])
            if not page_orders:
                print(f"No orders found on page {page}, ending pagination")
                break
                
            all_orders.extend(page_orders)
            print(f"Fetched {len(page_orders)} orders from page {page}")
            
            # Check for nextUrl to continue pagination
            next_url = data["data"].get("nextUrl")
            if not next_url:
                print(f"No nextUrl found, ending pagination at page {page}")
                break
            
            # Update URL for next request
            # nextUrl is usually a relative path, so prepend base_url
            try:
                if next_url.startswith('/'):
                    url = f"{base_url}{next_url}"
                elif next_url.startswith('http'):
                    url = next_url
                else:
                    url = f"{base_url}/{next_url.lstrip('/')}"
            except Exception as e:
                print(f"Error processing nextUrl '{next_url}': {e}, ending pagination")
                break
            
            # Clear params for subsequent requests since nextUrl contains all needed parameters
            params = {}
            page += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching orders on page {page}: {e}")
            break
    
    print(f"Total orders fetched for window {start_date} to {end_date}: {len(all_orders)}")
    return all_orders


def get_all_orders(start_date: str, end_date: str) -> List[Dict]:
    """
    Fetch orders from EasyEcom API with date windowing support
    
    IMPORTANT: This function ONLY accepts date range parameters.
    Do NOT pass filtering parameters like payment_mode, marketplace, etc.
    Those filters should be applied AFTER fetching using apply_filters().
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD HH:MM:SS'
        end_date: End date in format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
        List of order dictionaries (unfiltered)
    
    Example workflow for filtered data:
        1. orders = get_all_orders(start_date, end_date)  # Fetch all orders
        2. filtered = apply_filters(orders, [{"field": "payment_mode", "operator": "eq", "value": "PrePaid"}])
    """
    api_key = "54cdda5acf9d6b7df4d9a84b5a97ed0016417a0f"
    jwt_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczpcL1wvbG9hZGJhbGFuY2VyLXYyLW0uZWFzeWVjb20uaW9cL2FjY2Vzc1wvdG9rZW4iLCJpYXQiOjE3NzAxMjY3MzgsImV4cCI6MTc3ODAxMDczOCwibmJmIjoxNzcwMTI2NzM4LCJqdGkiOiJGaFZ0RXN3aGFuSEsyZmZ6Iiwic3ViIjoyNzA4MTUsInBydiI6ImE4NGRlZjY0YWQwMTE1ZDVlY2NjMWY4ODQ1YmNkMGU3ZmU2YzRiNjAiLCJ1c2VyX2lkIjoyNzA4MTUsImNvbXBhbnlfaWQiOjIyODc4MSwicm9sZV90eXBlX2lkIjoyLCJwaWlfYWNjZXNzIjowLCJwaWlfcmVwb3J0X2FjY2VzcyI6MCwicm9sZXMiOm51bGwsImNfaWQiOjIyODc4MSwidV9pZCI6MjcwODE1LCJsb2NhdGlvbl9yZXF1ZXN0ZWRfZm9yIjoyMjg3ODF9.D5lj-ntJW9HTGm9QIV6--xuiZY7EslaRICNGxGU7I_I"
    base_url = "https://api.easyecom.io"
    
    if not api_key or not jwt_token:
        raise ValueError("EASYECOM_API_KEY and EASYECOM_JWT_TOKEN must be set in .env")
    
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    
    # Calculate days difference
    days_diff = (end_dt - start_dt).days
    
    print(f"Date range: {start_date} to {end_date} (Days: {days_diff})")
    
    all_orders = []
    
    # If more than 7 days, implement windowing
    if days_diff > 7:
        print(f"Implementing windowing for {days_diff} days (> 7 days)")
        current_start = start_dt
        window_num = 1
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=7), end_dt)
            
            print(f"Fetching window {window_num}: {current_start} to {current_end}")
            
            # Fetch window with pagination
            window_orders = _fetch_orders_window(
                current_start.strftime("%Y-%m-%d %H:%M:%S"),
                current_end.strftime("%Y-%m-%d %H:%M:%S"),
                api_key,
                jwt_token,
                base_url
            )
            all_orders.extend(window_orders)
            print(f"Window {window_num} completed: {len(window_orders)} orders")
            
            # Move to next window
            current_start = current_end
            window_num += 1
    else:
        # Single fetch for <= 7 days with pagination
        print(f"Fetching all orders for single window ({days_diff} days <= 7)")
        all_orders = _fetch_orders_window(start_date, end_date, api_key, jwt_token, base_url)
    
    print(f"Total orders fetched across all windows/pages: {len(all_orders)}")
    return all_orders


if __name__ == "__main__":
    # Months from September 2025 to February 2026
    months = [
        (2026, 1), (2026, 2)
    ]
    
    for year, month in months:
        # Get the last day of the month
        _, last_day = calendar.monthrange(year, month)
        
        # Create month folder
        folder_name = f"{year}-{month:02d}"
        os.makedirs(folder_name, exist_ok=True)
        
        print(f"Processing month {folder_name}")
        
        for day in range(1, last_day + 1):
            # Start date: start of day
            start_date = datetime(year, month, day, 0, 0, 0)
            # End date: end of day
            end_date = datetime(year, month, day, 23, 59, 59)
            
            # Format to strings
            start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
            end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"Fetching orders for {year}-{month:02d}-{day:02d}")
            
            # Fetch orders
            orders = get_all_orders(start_date_str, end_date_str)
            
            # Save to JSON file in folder
            filename = f"{folder_name}/{year}-{month:02d}-{day:02d}.json"
            with open(filename, "w") as f:
                json.dump(orders, f, indent=4)
            
            print(f"Saved {len(orders)} orders to {filename}")
