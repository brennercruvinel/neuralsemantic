"""Simple test to verify pytest setup"""

def test_simple():
    """Test that pytest is working"""
    assert True

def test_math():
    """Test basic math operations"""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
    assert 10 / 2 == 5

def test_string_operations():
    """Test string operations"""
    text = "Neural Semantic Compiler"
    assert len(text) == 24
    assert text.lower() == "neural semantic compiler"
    assert text.split()[0] == "Neural"