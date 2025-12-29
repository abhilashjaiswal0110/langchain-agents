"""Comprehensive agent testing script to verify all fixes.

Tests all enterprise agents with API calls and verifies responses.
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_test(name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {status} - {name}")
    if details:
        print(f"  {details[:200]}...")

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        data = response.json()
        success = response.status_code == 200 and data.get("status") == "healthy"
        print_test("Health Check", success, f"Status: {data.get('status')}, Enterprise: {data.get('enterprise_agents_loaded')}")
        return success
    except Exception as e:
        print_test("Health Check", False, str(e))
        return False

def test_document_generator():
    """Test Document Generator Agent."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/enterprise/documents/invoke",
            json={"doc_type": "wli", "title": "Test WLI", "description": "Test procedure"},
            timeout=120
        )
        data = response.json()
        success = data.get("success") and len(data.get("response", "")) > 50
        print_test("Document Generator Agent", success, data.get("response", "")[:200])
        return success
    except Exception as e:
        print_test("Document Generator Agent", False, str(e))
        return False

def test_research_agent():
    """Test Research Agent."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/enterprise/research/invoke",
            json={"query": "What is Python?"},
            timeout=120
        )
        data = response.json()
        success = data.get("success") and len(data.get("response", "")) > 50
        print_test("Research Agent", success, data.get("response", "")[:200])
        return success
    except Exception as e:
        print_test("Research Agent", False, str(e))
        return False

def test_hitl_support():
    """Test HITL Support Agent."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/enterprise/support/invoke",
            json={"message": "I need help with VPN", "user_id": "test-user"},
            timeout=120
        )
        data = response.json()
        success = data.get("success") and len(data.get("response", "")) > 10
        print_test("HITL Support Agent", success, data.get("response", "")[:200])
        return success
    except Exception as e:
        print_test("HITL Support Agent", False, str(e))
        return False

def test_code_assistant():
    """Test Code Assistant Agent."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/enterprise/code/invoke",
            json={"code": "def hello():\n    print('hello')", "language": "python", "action": "analyze"},
            timeout=120
        )
        data = response.json()
        success = data.get("success") and len(data.get("response", "")) > 50
        print_test("Code Assistant Agent", success, data.get("response", "")[:200])
        return success
    except Exception as e:
        print_test("Code Assistant Agent", False, str(e))
        return False

def test_multilingual_rag():
    """Test Multilingual RAG Agent."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/enterprise/rag/invoke",
            json={"query": "What documents are available?"},
            timeout=60
        )
        data = response.json()
        success = data.get("success") and len(data.get("response", "")) > 10
        print_test("Multilingual RAG Agent", success, data.get("response", "")[:200])
        return success
    except Exception as e:
        print_test("Multilingual RAG Agent", False, str(e))
        return False

def test_content_agent():
    """Test Content Agent."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/enterprise/content/invoke",
            json={"topic": "AI trends", "platform": "linkedin", "tone": "professional", "audience": "tech"},
            timeout=120
        )
        data = response.json()
        success = data.get("success") and len(data.get("response", "")) > 50
        print_test("Content Agent", success, data.get("response", "")[:200])
        return success
    except Exception as e:
        print_test("Content Agent", False, str(e))
        return False

def test_data_analyst():
    """Test Data Analyst Agent."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/enterprise/data-analyst/invoke",
            json={"message": "What can you analyze?"},
            timeout=60
        )
        data = response.json()
        success = data.get("success") and len(data.get("response", "")) > 10
        print_test("Data Analyst Agent", success, data.get("response", "")[:200])
        return success
    except Exception as e:
        print_test("Data Analyst Agent", False, str(e))
        return False

def main():
    print("=" * 70)
    print("COMPREHENSIVE AGENT TESTING - All Enterprise Agents")
    print("=" * 70)

    tests = [
        ("Health", test_health),
        ("Document Generator", test_document_generator),
        ("Research", test_research_agent),
        ("HITL Support", test_hitl_support),
        ("Code Assistant", test_code_assistant),
        ("Multilingual RAG", test_multilingual_rag),
        ("Content", test_content_agent),
        ("Data Analyst", test_data_analyst),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
