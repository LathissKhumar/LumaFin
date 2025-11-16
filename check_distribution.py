#!/usr/bin/env python3
import psycopg2
from psycopg2.extras import RealDictCursor

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    database='lumafin_db',
    user='lumafin_user',
    password='lumafin_pass',
    port=5432
)
cur = conn.cursor(cursor_factory=RealDictCursor)

# Get category distribution
cur.execute("""
    SELECT 
        gt.category_name,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM global_examples), 2) as percentage
    FROM global_examples ge
    JOIN global_taxonomy gt ON ge.category_id = gt.id
    GROUP BY gt.category_name, gt.id
    ORDER BY count DESC
""")

results = cur.fetchall()

print("\n" + "=" * 70)
print("CATEGORY DISTRIBUTION AFTER RESEEDING".center(70))
print("=" * 70)

total = 0
uncategorized_pct = 0

for row in results:
    pct = row['percentage']
    count = row['count']
    category = row['category_name']
    print(f"  {category:25} │ {count:6,} │ {pct:7.2f}%")
    total += count
    if category == 'Uncategorized':
        uncategorized_pct = pct

print("=" * 70)
print(f"  {'TOTAL':25} │ {total:6,} │ {100.00:7.2f}%")
print("=" * 70)

proper_categorization = 100 - uncategorized_pct

print(f"\n📊 Key Metrics:")
print(f"  • Total Examples: {total:,}")
print(f"  • Uncategorized: {uncategorized_pct:.2f}%")
print(f"  • Properly Categorized: {proper_categorization:.2f}%")

if proper_categorization >= 90:
    print(f"\n✅ TARGET ACHIEVED: {proper_categorization:.2f}% ≥ 90% ✅")
    print("✅ All further work can proceed ✅")
elif proper_categorization >= 85:
    print(f"\n⚠️  ACCEPTABLE: {proper_categorization:.2f}% (target 90%, gap: {90 - proper_categorization:.2f}%)")
else:
    print(f"\n❌ NEEDS IMPROVEMENT: {proper_categorization:.2f}% (target 90%, gap: {90 - proper_categorization:.2f}%)")

cur.close()
conn.close()
