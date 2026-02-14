"""Sample data for RAG testing."""

# Sample sales data (CSV format)
SAMPLE_SALES_CSV = """
product_name,product_id,quarter,sales_amount,units_sold,region
iPhone 15 Pro,PROD001,Q4 2024,450000,1500,North America
iPhone 15 Pro,PROD001,Q3 2024,380000,1200,North America
MacBook Pro 16,PROD002,Q4 2024,650000,800,North America
MacBook Air M3,PROD003,Q4 2024,420000,1200,Europe
iPad Air,PROD004,Q4 2024,280000,1500,Asia Pacific
Apple Watch Series 9,PROD005,Q4 2024,95000,3500,North America
AirPods Pro 2,PROD006,Q4 2024,180000,4500,Global
"""

# Sample customer feedback
SAMPLE_FEEDBACK_MD = """
# Customer Feedback Summary - Q4 2024

## Product Performance

### iPhone 15 Pro
- Average rating: 4.8/5
- Top praise: "Excellent camera quality", "Fast performance", "Beautiful design"
- Main complaint: "High price point", "Limited customization"
- Sentiment: Positive (89% positive reviews)

### MacBook Pro 16
- Average rating: 4.7/5
- Top praise: "Powerful performance", "Beautiful display", "Great build quality"
- Main complaint: "Expensive", "Fan noise under load"
- Sentiment: Very Positive (92% positive reviews)

### iPad Air
- Average rating: 4.6/5
- Top praise: "Great value", "Versatile OS", "Good performance"
- Main complaint: "Lack of USB-C", "Battery drain with heavy apps"
- Sentiment: Positive (85% positive reviews)

## Regional Insights

### North America
- Strongest region for premium products
- High demand for MacBook Pro and iPhone 15 Pro
- Average order value: $1,250

### Europe
- Growing iPad market
- Preference for wireless accessories
- Average order value: $980

### Asia Pacific
- Mobile-first market
- Strong AirPods demand
- Growing laptop market
- Average order value: $750

## Overall Trends
1. Premium products gaining market share
2. Customers value ecosystem integration
3. Environmental concerns rising - want recyclable packaging
4. Support quality most important factor after price
"""
