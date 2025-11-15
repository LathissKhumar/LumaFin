from src.preprocessing.normalize import normalize_merchant, bucket_amount

def test_normalize_merchant():
    assert normalize_merchant("SQ *COFFEE SHOP") == "square coffee shop"
    assert normalize_merchant("AMZN Mktp US") == "amazon mktp us"
    assert normalize_merchant("Starbucks #1234") == "starbucks"
    
def test_bucket_amount():
    assert bucket_amount(5.99) == "0-10"
    assert bucket_amount(25.00) == "10-50"
    assert bucket_amount(150.00) == "100-500"

if __name__ == "__main__":
    test_normalize_merchant()
    test_bucket_amount()
    print("✓ All preprocessing tests passed!")
