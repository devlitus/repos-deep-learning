#!/bin/bash
cd /d/work/repos-deep-learning/recommendation-fashion

# Create test dataset with 1000 sample reviews
echo "Generating test dataset..."

python3 << 'PYTHON'
import json
import random
from pathlib import Path

Path('data/raw').mkdir(parents=True, exist_ok=True)

product_prefixes = ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'hat', 'sweater', 'jeans', 'coat', 'blouse']
summaries = [
    'Excellent product!',
    'Great quality and fast shipping',
    'Very satisfied',
    'Love it!',
    'Worth the price',
    'Poor quality',
    'Not as described',
    'Disappointing',
    'Average quality',
    'Good value for money'
]

with open('data/raw/fashion_reviews.json', 'w', encoding='utf-8') as f:
    for i in range(1000):
        user_id = f'A{random.randint(1000, 1500):05d}'
        product_id = f'B{random.randint(0, 800):08d}'
        rating = random.choices([1, 2, 3, 4, 5], weights=[10, 15, 25, 25, 25])[0]
        timestamp = random.randint(1400000000, 1600000000)
        
        review = {
            'reviewerID': user_id,
            'asin': product_id,
            'overall': rating,
            'summary': random.choice(summaries),
            'reviewText': f'This {random.choice(product_prefixes)} is ' +
                         ('great!' if rating >= 4 else 'not great.' if rating <= 2 else 'okay.'),
            'unixReviewTime': timestamp
        }
        f.write(json.dumps(review) + '\n')
        
        if (i + 1) % 200 == 0:
            print(f"Generated {i+1}/1000 reviews...")

print("✅ Test dataset created successfully!")
file_size = Path('data/raw/fashion_reviews.json').stat().st_size
print(f"File size: {file_size / 1024:.2f} KB")
PYTHON

