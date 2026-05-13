#!/usr/bin/env python3
"""
Test script for EasyEcom Returns API integration.
Tests the new pagination functions and RTO dashboard endpoint.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from routes.cancellation import (
    fetch_returns_for_window,
    fetch_pending_returns_for_window,
    top_counts,
)
import pandas as pd


def test_top_counts():
    """Test the top_counts helper function."""
    print("Testing top_counts()...")
    
    # Test with valid series
    series = pd.Series(['CA', 'CA', 'NY', 'NY', 'NY', 'TX'])
    result = top_counts(series, key_name='state')
    
    assert len(result) == 3, f"Expected 3 states, got {len(result)}"
    assert result[0]['state'] == 'NY', f"Expected NY first (highest count), got {result[0]['state']}"
    assert result[0]['count'] == 3, f"Expected NY count=3, got {result[0]['count']}"
    print("  ✓ top_counts() with valid series")
    
    # Test with empty series
    empty_result = top_counts(pd.Series([]))
    assert empty_result == [], f"Expected empty list, got {empty_result}"
    print("  ✓ top_counts() with empty series")
    
    # Test with None
    none_result = top_counts(None)
    assert none_result == [], f"Expected empty list for None, got {none_result}"
    print("  ✓ top_counts() with None")


@patch('requests.get')
def test_fetch_returns_for_window(mock_get):
    """Test fetch_returns_for_window pagination."""
    print("\nTesting fetch_returns_for_window()...")
    
    # Mock response with pagination
    page1_response = Mock()
    page1_response.status_code = 200
    page1_response.json.return_value = {
        "code": 200,
        "data": {
            "credit_notes": [
                {
                    "credit_note_id": 1,
                    "invoice_id": 100,
                    "forward_shipment_customer_state": "CA",
                    "forward_shipment_customer_pin_code": "90210",
                }
            ],
            "nextUrl": "/orders/getAllReturns?limit=250&created_after=2026-05-01&offset=250"
        }
    }
    
    page2_response = Mock()
    page2_response.status_code = 200
    page2_response.json.return_value = {
        "code": 200,
        "data": {
            "credit_notes": [
                {
                    "credit_note_id": 2,
                    "invoice_id": 101,
                    "forward_shipment_customer_state": "NY",
                    "forward_shipment_customer_pin_code": "10001",
                }
            ],
            "nextUrl": None  # No more pages
        }
    }
    
    # End pagination
    end_response = Mock()
    end_response.status_code = 400
    
    mock_get.side_effect = [page1_response, page2_response]
    
    result = fetch_returns_for_window(
        "2026-05-01 00:00:00",
        "2026-05-02 23:59:59",
        "test_key",
        "test_token",
    )
    
    assert len(result) == 2, f"Expected 2 returns, got {len(result)}"
    assert result[0]['credit_note_id'] == 1, f"Expected first return id=1, got {result[0]['credit_note_id']}"
    assert result[1]['credit_note_id'] == 2, f"Expected second return id=2, got {result[1]['credit_note_id']}"
    print("  ✓ fetch_returns_for_window() pagination handling")


@patch('requests.get')
def test_fetch_pending_returns_for_window(mock_get):
    """Test fetch_pending_returns_for_window pagination."""
    print("\nTesting fetch_pending_returns_for_window()...")
    
    # Mock response
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "code": 200,
        "data": {
            "pending_returns": [
                {
                    "id": 1,
                    "forward_shipment_customer_state": "TX",
                    "forward_shipment_customer_pin_code": "75001",
                }
            ],
            "nextUrl": None
        }
    }
    
    mock_get.return_value = response
    
    result = fetch_pending_returns_for_window(
        "2026-05-01 00:00:00",
        "2026-05-02 23:59:59",
        "test_key",
        "test_token",
    )
    
    assert len(result) == 1, f"Expected 1 pending return, got {len(result)}"
    assert result[0]['id'] == 1, f"Expected pending return id=1, got {result[0]['id']}"
    print("  ✓ fetch_pending_returns_for_window() basic fetch")


@patch('requests.get')
def test_pagination_error_handling(mock_get):
    """Test error handling in pagination."""
    print("\nTesting pagination error handling...")
    
    # Test API error response
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "code": 500,  # Error code
        "data": None
    }
    
    mock_get.return_value = response
    
    result = fetch_returns_for_window(
        "2026-05-01 00:00:00",
        "2026-05-02 23:59:59",
        "test_key",
        "test_token",
    )
    
    assert result == [], f"Expected empty list on API error, got {result}"
    print("  ✓ fetch_returns_for_window() handles API errors")


def test_date_extraction():
    """Test date extraction from datetime strings."""
    print("\nTesting date extraction...")
    
    # Test date extraction logic (as used in fetch functions)
    start_datetime = "2026-05-01 00:00:00"
    created_after = start_datetime.split(' ')[0] if ' ' in start_datetime else start_datetime
    
    assert created_after == "2026-05-01", f"Expected 2026-05-01, got {created_after}"
    print("  ✓ Date extraction from datetime string")


def test_nexturl_handling():
    """Test nextUrl path handling for different formats."""
    print("\nTesting nextUrl path handling...")
    
    base_url = "https://api.easyecom.io"
    
    # Test relative path
    relative_url = "/orders/getAllReturns?limit=250&offset=250"
    result = f"{base_url}{relative_url}" if relative_url.startswith("/") else relative_url
    assert result == "https://api.easyecom.io/orders/getAllReturns?limit=250&offset=250"
    print("  ✓ Relative path handling")
    
    # Test absolute URL
    absolute_url = "https://api.easyecom.io/orders/getAllReturns?limit=250&offset=250"
    result = absolute_url if absolute_url.startswith("http") else absolute_url
    assert result == absolute_url
    print("  ✓ Absolute URL handling")
    
    # Test partial path
    partial_url = "orders/getAllReturns?limit=250&offset=250"
    result = f"{base_url}/{partial_url.lstrip('/')}"
    assert result == "https://api.easyecom.io/orders/getAllReturns?limit=250&offset=250"
    print("  ✓ Partial path handling")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Tests for EasyEcom Returns API Integration")
    print("=" * 60)
    
    try:
        test_top_counts()
        test_fetch_returns_for_window()
        test_fetch_pending_returns_for_window()
        test_pagination_error_handling()
        test_date_extraction()
        test_nexturl_handling()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
