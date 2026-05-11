#!/usr/bin/env python3
"""Export DynamoDB table to CSV with proper type handling."""

import argparse
import csv
import json
import sys
from typing import Any, Dict, List

import boto3


def deserialize_dynamodb_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert DynamoDB item format to Python dict."""
    result = {}
    for key, value in item.items():
        if isinstance(value, dict):
            if 'S' in value:
                result[key] = value['S']
            elif 'N' in value:
                result[key] = float(value['N']) if '.' in value['N'] else int(value['N'])
            elif 'B' in value:
                result[key] = value['B']
            elif 'SS' in value:
                result[key] = ', '.join(value['SS'])
            elif 'NS' in value:
                result[key] = ', '.join(value['NS'])
            elif 'BS' in value:
                result[key] = ', '.join(value['BS'])
            elif 'M' in value:
                result[key] = json.dumps(value['M'])
            elif 'L' in value:
                result[key] = json.dumps(value['L'])
            elif 'NULL' in value:
                result[key] = None
            elif 'BOOL' in value:
                result[key] = value['BOOL']
            else:
                result[key] = str(value)
        else:
            result[key] = value
    return result


def export_table_to_csv(
    table_name: str,
    output_file: str,
    region: str = "ap-south-1",
    limit: int = None
) -> int:
    """Scan DynamoDB table and export to CSV."""
    dynamodb = boto3.client('dynamodb', region_name=region)
    
    print(f"Scanning DynamoDB table: {table_name}")
    print(f"Region: {region}")
    print(f"Output file: {output_file}")
    
    all_items = []
    last_evaluated_key = None
    scan_count = 0
    
    try:
        while True:
            kwargs = {
                'TableName': table_name,
                'Limit': limit or 100
            }
            
            if last_evaluated_key:
                kwargs['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.scan(**kwargs)
            items = response.get('Items', [])
            
            # Deserialize items
            for item in items:
                all_items.append(deserialize_dynamodb_item(item))
            
            scan_count += len(items)
            print(f"  Scanned {scan_count} items...", end='\r')
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
    
    except Exception as e:
        print(f"Error scanning table: {e}")
        return 0
    
    print(f"\nTotal items scanned: {scan_count}")
    
    if not all_items:
        print("No items found in table")
        return 0
    
    # Get all unique keys across all items
    all_keys = set()
    for item in all_items:
        all_keys.update(item.keys())
    
    all_keys = sorted(list(all_keys))
    
    # Write to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(all_items)
        
        print(f"✓ Export successful!")
        print(f"  Rows exported: {len(all_items)}")
        print(f"  Columns: {len(all_keys)}")
        print(f"  File: {output_file}")
        return len(all_items)
    
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Export DynamoDB table to CSV")
    parser.add_argument('--table', required=True, help="DynamoDB table name")
    parser.add_argument('--region', default='ap-south-1', help="AWS region (default: ap-south-1)")
    parser.add_argument('--output', help="Output CSV file (default: {table}_export.csv)")
    parser.add_argument('--limit', type=int, help="Scan limit per request (default: 100)")
    
    args = parser.parse_args()
    
    output_file = args.output or f"{args.table}_export.csv"
    
    count = export_table_to_csv(
        table_name=args.table,
        output_file=output_file,
        region=args.region,
        limit=args.limit
    )
    
    sys.exit(0 if count > 0 else 1)


if __name__ == '__main__':
    main()
