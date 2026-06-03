ALL_GENERATED_TOOLS = {
  "get_all_orders": {
    "name": "get_all_orders",
    "description": "Fetch and aggregate daily orders from S3 for a date range.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "start_date": {
          "type": "STRING",
          "description": "Date in format 'YYYY-MM-DD HH:MM:SS'"
        },
        "end_date": {
          "type": "STRING",
          "description": "Date in format 'YYYY-MM-DD HH:MM:SS'"
        }
      },
      "required": [
        "start_date",
        "end_date"
      ]
    }
  },
  "convert_to_df": {
    "name": "convert_to_df",
    "description": "Convert raw JSON order data to normalized DataFrame with optimized chunking",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "raw": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "raw"
      ]
    }
  },
  "get_aov": {
    "name": "get_aov",
    "description": "Calculate Average Order Value from orders DataFrame",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_total_revenue": {
    "name": "get_total_revenue",
    "description": "Calculate total revenue from orders DataFrame",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_order_count": {
    "name": "get_order_count",
    "description": "Get total number of orders",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_order_status_distribution": {
    "name": "get_order_status_distribution",
    "description": "Get distribution of order statuses",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_payment_mode_distribution": {
    "name": "get_payment_mode_distribution",
    "description": "Get distribution of payment modes",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_marketplace_distribution": {
    "name": "get_marketplace_distribution",
    "description": "Get distribution of orders by marketplace",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_state_wise_distribution": {
    "name": "get_state_wise_distribution",
    "description": "Get distribution of orders by state",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_city_wise_distribution": {
    "name": "get_city_wise_distribution",
    "description": "Get distribution of orders by city (top N cities)",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "top_n": {
          "type": "INTEGER",
          "description": "Number of top records to return"
        }
      },
      "required": [
        "table",
        "top_n"
      ]
    }
  },
  "get_courier_distribution": {
    "name": "get_courier_distribution",
    "description": "Get distribution of orders by courier service",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_average_discount": {
    "name": "get_average_discount",
    "description": "Calculate average discount amount",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_average_shipping_charge": {
    "name": "get_average_shipping_charge",
    "description": "Calculate average shipping charge",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_average_tax": {
    "name": "get_average_tax",
    "description": "Calculate average tax amount",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_conversion_rate": {
    "name": "get_conversion_rate",
    "description": "Calculate order conversion rate based on successful deliveries",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "success_status": {
          "type": "STRING",
          "description": "Parameter success_status"
        }
      },
      "required": [
        "table",
        "success_status"
      ]
    }
  },
  "get_cod_vs_prepaid_metrics": {
    "name": "get_cod_vs_prepaid_metrics",
    "description": "Compare COD vs PrePaid payment methods",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_common_metrics": {
    "name": "get_common_metrics",
    "description": "Calculate common business metrics from order data",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "data": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "data"
      ]
    }
  },
  "get_geographic_insights": {
    "name": "get_geographic_insights",
    "description": "Get geographic distribution insights",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "top_n": {
          "type": "INTEGER",
          "description": "Number of top records to return"
        }
      },
      "required": [
        "table",
        "top_n"
      ]
    }
  },
  "get_schema_info": {
    "name": "get_schema_info",
    "description": "Return schema and metadata information about available data entities",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "entity": {
          "type": "STRING",
          "description": "Parameter entity"
        },
        "field": {
          "type": "STRING",
          "description": "Field name to analyze"
        }
      },
      "required": []
    }
  },
  "get_cancelled_count": {
    "name": "get_cancelled_count",
    "description": "Return count cancelled orders",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_vendor_cost_sheet": {
    "name": "get_vendor_cost_sheet",
    "description": "Fetch vendor_cost_sheet from Supabase & return as List[Dict].",
    "parameters": {
      "type": "OBJECT",
      "properties": {},
      "required": []
    }
  },
  "get_margin": {
    "name": "get_margin",
    "description": "Calculates row-wise Margin % and returns the sum.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_gross_profit": {
    "name": "get_gross_profit",
    "description": "Returns the total sum of Gross Profit across all items.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_markup": {
    "name": "get_markup",
    "description": "Calculates row-wise Markup and returns the total sum.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_selling_price": {
    "name": "get_selling_price",
    "description": "Returns total sum of MRP.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_cost_price": {
    "name": "get_cost_price",
    "description": "Returns total sum of Final price.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_cost_to_price_ratio": {
    "name": "get_cost_to_price_ratio",
    "description": "Cost-to-Price Ratio % = (Total Cost / Total Selling Price) * 100",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "execute_custom_calculation": {
    "name": "execute_custom_calculation",
    "description": "Safe execution of LLM-generated Python code for custom metrics on a DataFrame. Use this as a FALLBACK if no specific tool exists. Example: Finding highest selling SKU, filtering on nested fields, or complex aggregations. The DataFrame 'df' has exploded suborders (fields prefixed with 'suborder_'). You MUST assign the final answer to the 'result' variable. Available modules: pd, np, math, datetime.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the DataFrame (usually ends in '_df' or '_filtered_df')"
        },
        "calculation_code": {
          "type": "STRING",
          "description": "Python code. Example: result = df['suborder_sku'].value_counts().idxmax()"
        },
        "metric_name": {
          "type": "STRING",
          "description": "Descriptive name for the calculation"
        }
      },
      "required": [
        "table",
        "calculation_code",
        "metric_name"
      ]
    }
  },
  "apply_filters": {
    "name": "apply_filters",
    "description": "Apply filters to order table (ALSO WORKS FOR PROFIT TABLE, SINCE NO COMMON PARAMS)",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "filters": {
          "type": "ARRAY",
          "items": {
            "type": "OBJECT"
          },
          "description": "List of filter objects: [{'field': '...', 'operator': '...', 'value': '...'}]"
        }
      },
      "required": [
        "table",
        "filters"
      ]
    }
  },
  "get_statistical_summary": {
    "name": "get_statistical_summary",
    "description": "Get comprehensive statistical summary for a numeric field",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "field": {
          "type": "STRING",
          "description": "Field name to analyze"
        }
      },
      "required": [
        "table",
        "field"
      ]
    }
  },
  "get_percentile": {
    "name": "get_percentile",
    "description": "Get specific percentile for a numeric field",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "field": {
          "type": "STRING",
          "description": "Field name to analyze"
        },
        "percentile": {
          "type": "NUMBER",
          "description": "Percentile value (0-100)"
        }
      },
      "required": [
        "table",
        "field",
        "percentile"
      ]
    }
  },
  "get_top_percentile": {
    "name": "get_top_percentile",
    "description": "Get records in top percentile for a field",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "field": {
          "type": "STRING",
          "description": "Field name to analyze"
        },
        "percentile": {
          "type": "NUMBER",
          "description": "Percentile value (0-100)"
        }
      },
      "required": [
        "table",
        "field",
        "percentile"
      ]
    }
  },
  "get_bottom_percentile": {
    "name": "get_bottom_percentile",
    "description": "Get records in bottom percentile for a field",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "field": {
          "type": "STRING",
          "description": "Field name to analyze"
        },
        "percentile": {
          "type": "NUMBER",
          "description": "Percentile value (0-100)"
        }
      },
      "required": [
        "table",
        "field",
        "percentile"
      ]
    }
  },
  "get_correlation_matrix": {
    "name": "get_correlation_matrix",
    "description": "Calculate correlation matrix between numeric fields",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "fields": {
          "type": "ARRAY",
          "items": {
            "type": "STRING"
          },
          "description": "List of field names"
        }
      },
      "required": [
        "table",
        "fields"
      ]
    }
  },
  "get_payment_cycle_data": {
    "name": "get_payment_cycle_data",
    "description": "Fetch payment cycle and cash discount data from Supabase.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "distributor_name": {
          "type": "STRING",
          "description": "Optional distributor name to filter by"
        }
      },
      "required": []
    }
  },
  "get_avg_margin": {
    "name": "get_avg_margin",
    "description": "Calculate average margin across all distributors.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_weighted_avg_margin": {
    "name": "get_weighted_avg_margin",
    "description": "Calculate weighted average margin using sales volume if available.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "volume_column": {
          "type": "STRING",
          "description": "Parameter volume_column"
        }
      },
      "required": [
        "table",
        "volume_column"
      ]
    }
  },
  "get_margin_per_payment_day": {
    "name": "get_margin_per_payment_day",
    "description": "Calculate efficiency score: Average Margin per Payment Day.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_total_margin_exposure": {
    "name": "get_total_margin_exposure",
    "description": "Calculate total margin exposure: SUM(margin \u00d7 estimated_monthly_sales).",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "avg_monthly_sales_column": {
          "type": "STRING",
          "description": "Parameter avg_monthly_sales_column"
        }
      },
      "required": [
        "table",
        "avg_monthly_sales_column"
      ]
    }
  },
  "get_high_risk_distributors": {
    "name": "get_high_risk_distributors",
    "description": "Identify high-risk distributors: margin > threshold AND payment_cycle > threshold.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        },
        "margin_threshold": {
          "type": "STRING",
          "description": "Parameter margin_threshold"
        },
        "cycle_threshold": {
          "type": "STRING",
          "description": "Parameter cycle_threshold"
        }
      },
      "required": [
        "table",
        "margin_threshold",
        "cycle_threshold"
      ]
    }
  },
  "get_cycle_efficiency_score": {
    "name": "get_cycle_efficiency_score",
    "description": "Calculate cycle efficiency score for portfolio.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_payment_cycle_distribution": {
    "name": "get_payment_cycle_distribution",
    "description": "Get distribution of payment cycles (grouped by day ranges).",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  },
  "get_cash_discount_stats": {
    "name": "get_cash_discount_stats",
    "description": "Calculate statistics for cash discount (CD) field.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "table": {
          "type": "STRING",
          "description": "Reference ID for the data source (e.g., from fetch tools)"
        }
      },
      "required": [
        "table"
      ]
    }
  }
}
