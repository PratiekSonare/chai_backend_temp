"""
Data Schema Definitions for Chupps Pipeline
============================================

This module defines the schema for all data sources used in the pipeline.
Use this to:
1. Validate filters before execution
2. Provide field hints to the LLM
3. Generate schema discovery responses
4. Catch filter errors early with helpful messages
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# ==========================================
# ORDER FIELD SCHEMA
# ==========================================

@dataclass
class FieldDefinition:
    """Defines a single field in a data source"""
    name: str
    description: str
    data_type: str  # 'string', 'number', 'date', 'array'
    examples: List[Any]
    required: bool = False
    nested_in: Optional[str] = None  # 'suborders' if nested in suborder array


# Order-level fields (top-level in order document)
ORDERS_TOP_LEVEL_FIELDS = {
    "order_id": FieldDefinition(
        name="order_id",
        description="Unique order identifier",
        data_type="string",
        examples=["ORD-2026-001", "ORD-2026-002"],
        required=True
    ),
    "invoice_id": FieldDefinition(
        name="invoice_id",
        description="Invoice number for the order",
        data_type="string",
        examples=["INV-001", "INV-002"]
    ),
    "order_date": FieldDefinition(
        name="order_date",
        description="Date when order was placed",
        data_type="date",
        examples=["2026-01-15", "2026-01-20"]
    ),
    "delivery_date": FieldDefinition(
        name="delivery_date",
        description="Date when order was delivered",
        data_type="date",
        examples=["2026-01-20", "2026-01-22"]
    ),
    "order_status": FieldDefinition(
        name="order_status",
        description="Current status of the order",
        data_type="string",
        examples=["Delivered", "Cancelled", "Returned", "Processing", "Open", "Shipped", "Manifest Scanned"],
        required=True
    ),
    "payment_mode": FieldDefinition(
        name="payment_mode",
        description="Payment method used for the order",
        data_type="string",
        examples=["PrePaid", "COD", "Credit Card"],
        required=True
    ),
    "marketplace": FieldDefinition(
        name="marketplace",
        description="E-commerce marketplace where order came from",
        data_type="string",
        examples=["Amazon", "Flipkart", "Direct", "Website", "Myntra PPMP", "Shopify"],
        required=True
    ),
    "state": FieldDefinition(
        name="state",
        description="Delivery state/province",
        data_type="string",
        examples=["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Telangana", "Uttar Pradesh"]
    ),
    "city": FieldDefinition(
        name="city",
        description="Delivery city",
        data_type="string",
        examples=["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Noida"],
        required=True
    ),
    "pin_code": FieldDefinition(
        name="pin_code",
        description="Postal code of delivery address",
        data_type="string",
        examples=["400001", "110001", "500097", "201304"]
    ),
    "customer_name": FieldDefinition(
        name="customer_name",
        description="Name of the customer",
        data_type="string",
        examples=["Rajesh Kumar", "Priya Singh"]
    ),
    "total_amount": FieldDefinition(
        name="total_amount",
        description="Total order value (including taxes, excluding discount)",
        data_type="number",
        examples=[500, 1500, 5000]
    ),
    "total_discount": FieldDefinition(
        name="total_discount",
        description="Total discount amount",
        data_type="number",
        examples=[50, 200, 500]
    ),
    "total_tax": FieldDefinition(
        name="total_tax",
        description="Total tax amount (GST, etc.)",
        data_type="number",
        examples=[50, 100, 250]
    ),
    "total_shipping_charge": FieldDefinition(
        name="total_shipping_charge",
        description="Shipping/delivery charges",
        data_type="number",
        examples=[40, 60, 100]
    ),
    "order_quantity": FieldDefinition(
        name="order_quantity",
        description="Total number of items in the order",
        data_type="number",
        examples=[1, 2, 5]
    ),
    "courier": FieldDefinition(
        name="courier",
        description="Courier/shipping partner name",
        data_type="string",
        examples=["Delhivery", "FedEx", "myntra", "eKart", "Amazon Easyship"]
    ),
    "billing_state": FieldDefinition(
        name="billing_state",
        description="State of the billing address",
        data_type="string",
        examples=["Telangana", "Uttar Pradesh", "Maharashtra"]
    ),
    "billing_city": FieldDefinition(
        name="billing_city",
        description="City of the billing address",
        data_type="string",
        examples=["Hyderabad", "Bareilly", "Mumbai"]
    ),
    "order_type": FieldDefinition(
        name="order_type",
        description="Type of order (B2C, B2B)",
        data_type="string",
        examples=["B2C", "B2B"]
    ),
}

# Nested fields (inside suborders array)
ORDERS_NESTED_FIELDS = {
    "sku": FieldDefinition(
        name="sku",
        description="Product SKU (inside suborder)",
        data_type="string",
        examples=["CHUPPS-SHOE-001", "CHUPPS-BOOT-002"],
        nested_in="suborders"
    ),
    "brand": FieldDefinition(
        name="brand",
        description="Product brand (inside suborder)",
        data_type="string",
        examples=["Chupps", "Nike", "Adidas"],
        nested_in="suborders"
    ),
    "category": FieldDefinition(
        name="category",
        description="Product category (inside suborder)",
        data_type="string",
        examples=["Shoes", "Boots", "Sandals"],
        nested_in="suborders"
    ),
    "productName": FieldDefinition(
        name="productName",
        description="Product name (inside suborder)",
        data_type="string",
        examples=["Casual Leather Shoe", "Winter Boot"],
        nested_in="suborders"
    ),
    "selling_price": FieldDefinition(
        name="selling_price",
        description="Selling price per item (inside suborder)",
        data_type="number",
        examples=[500, 1200, 2500],
        nested_in="suborders"
    ),
    "mrp": FieldDefinition(
        name="mrp",
        description="Maximum Retail Price (inside suborder)",
        data_type="number",
        examples=[1000, 2000, 5000],
        nested_in="suborders"
    ),
    "item_quantity": FieldDefinition(
        name="item_quantity",
        description="Quantity of this item (inside suborder)",
        data_type="number",
        examples=[1, 2, 5],
        nested_in="suborders"
    ),
    "item_status": FieldDefinition(
        name="item_status",
        description="Status of individual item (inside suborder)",
        data_type="string",
        examples=["Delivered", "Cancelled", "Returned"],
        nested_in="suborders"
    ),
    "size": FieldDefinition(
        name="size",
        description="Product size (inside suborder)",
        data_type="string",
        examples=["7", "8", "9", "10", "M", "L", "XL"],
        nested_in="suborders"
    ),
    "cost": FieldDefinition(
        name="cost",
        description="Cost price (inside suborder)",
        data_type="number",
        examples=[300, 700, 1500],
        nested_in="suborders"
    ),
    "tax": FieldDefinition(
        name="tax",
        description="Tax on item (inside suborder)",
        data_type="number",
        examples=[45, 100, 250],
        nested_in="suborders"
    ),
    "weight": FieldDefinition(
        name="weight",
        description="Weight of item in kg (inside suborder)",
        data_type="number",
        examples=[0.5, 1.0, 2.0],
        nested_in="suborders"
    ),
}

# Combine all fields
ALL_ORDERS_FIELDS = {**ORDERS_TOP_LEVEL_FIELDS, **ORDERS_NESTED_FIELDS}

# ==========================================
# SCHEMA REGISTRY
# ==========================================

SCHEMA_REGISTRY = {
    "orders": {
        "description": "E-commerce order records",
        "top_level_fields": ORDERS_TOP_LEVEL_FIELDS,
        "nested_fields": ORDERS_NESTED_FIELDS,
        "all_fields": ALL_ORDERS_FIELDS,
        "primary_key": "order_id",
        "nested_arrays": ["suborders"]
    }
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_schema(data_source: str) -> Dict[str, Any]:
    """Get schema definition for a data source"""
    return SCHEMA_REGISTRY.get(data_source, {})


def get_available_fields(data_source: str) -> List[str]:
    """Get list of available fields for a data source"""
    schema = get_schema(data_source)
    return list(schema.get("all_fields", {}).keys())


def get_field_definition(data_source: str, field_name: str) -> Optional[FieldDefinition]:
    """Get definition for a specific field"""
    schema = get_schema(data_source)
    return schema.get("all_fields", {}).get(field_name)


def validate_filter_field(data_source: str, field_name: str) -> tuple[bool, str]:
    """
    Validate if a field exists in the data source.
    Returns: (is_valid, message)
    """
    schema = get_schema(data_source)
    available_fields = schema.get("all_fields", {})
    
    if field_name in available_fields:
        return True, f"✓ Field '{field_name}' is valid"
    
    # If not found, suggest similar fields
    import difflib
    suggestions = difflib.get_close_matches(field_name, available_fields, n=3, cutoff=0.6)
    
    if suggestions:
        return False, f"✗ Field '{field_name}' not found. Did you mean: {', '.join(suggestions)}?"
    else:
        return False, f"✗ Field '{field_name}' not found in {data_source} schema"


def get_schema_info(data_source: str) -> Dict[str, Any]:
    """
    Get detailed schema information formatted for the user/LLM.
    Useful for schema_discovery queries.
    """
    schema = get_schema(data_source)
    
    if not schema:
        return {"error": f"No schema found for data source: {data_source}"}
    
    return {
        "data_source": data_source,
        "description": schema.get("description"),
        "top_level_fields": {
            name: {
                "description": field.description,
                "type": field.data_type,
                "examples": field.examples[:2]  # Limit examples
            }
            for name, field in schema.get("top_level_fields", {}).items()
        },
        "nested_fields": {
            name: {
                "description": field.description,
                "type": field.data_type,
                "examples": field.examples[:2],
                "nested_in": field.nested_in
            }
            for name, field in schema.get("nested_fields", {}).items()
        },
        "common_filters": {
            "by_status": "order_status = 'Delivered' | 'Cancelled' | 'Returned'",
            "by_payment": "payment_mode = 'PrePaid' | 'COD'",
            "by_marketplace": "marketplace = 'Amazon' | 'Flipkart' | 'Direct'",
            "by_city": "city = 'Mumbai' | 'Delhi' | 'Bangalore' etc.",
            "by_state": "state = 'Maharashtra' | 'Delhi' | 'Karnataka' etc."
        }
    }


def validate_filter_list(data_source: str, filters: List[Dict]) -> tuple[bool, List[str]]:
    """
    Validate a list of filters.
    Returns: (is_valid, [list of error messages])
    """
    errors = []
    schema = get_schema(data_source)
    
    if not schema:
        errors.append(f"Unknown data source: {data_source}")
        return False, errors
    
    for filter_spec in filters:
        field = filter_spec.get("field")
        operator = filter_spec.get("operator", "eq")
        
        if not field:
            errors.append("Filter missing 'field' key")
            continue
        
        is_valid, message = validate_filter_field(data_source, field)
        if not is_valid:
            errors.append(message)
    
    return len(errors) == 0, errors


# ==========================================
# LLM PROMPT HELPER
# ==========================================

def get_schema_prompt(data_source: str) -> str:
    """
    Generate a system prompt snippet describing available fields.
    Use this in the LLM system instructions.
    """
    schema = get_schema(data_source)
    
    if not schema:
        return f"Data source '{data_source}' not found."
    
    top_level_fields = schema.get("top_level_fields", {})
    nested_fields = schema.get("nested_fields", {})
    
    prompt = f"""
## {data_source.upper()} DATA SCHEMA

### Available Top-Level Fields (filter these at order level):
"""
    
    for field_name, field_def in list(top_level_fields.items())[:30]:  # Limit to 30 fields
        prompt += f"\n- `{field_name}`: {field_def.description}"
        if field_def.examples:
            prompt += f" (examples: {', '.join(str(e) for e in field_def.examples[:2])})"
    
    if len(top_level_fields) > 30:
        prompt += f"\n... and {len(top_level_fields) - 30} more fields"
    
    if nested_fields:
        prompt += f"\n\n### Available Nested Fields (inside suborders array, {len(nested_fields)} fields):"
        for field_name in list(nested_fields.keys())[:5]:
            prompt += f"\n- `{field_name}`: (inside suborders)"
        if len(nested_fields) > 5:
            prompt += f"\n... and {len(nested_fields) - 5} more nested fields"
    
    prompt += f"""

### Common Filters:
- Filter by city: filter_field="city", operator="eq", value="Mumbai"
- Filter by status: filter_field="order_status", operator="eq", value="Delivered"
- Filter by payment: filter_field="payment_mode", operator="eq", value="PrePaid"
- Filter by marketplace: filter_field="marketplace", operator="eq", value="Amazon"

IMPORTANT: Always use the EXACT field names above. For delivery city, use "city" not "delivery_city".
"""
    
    return prompt
