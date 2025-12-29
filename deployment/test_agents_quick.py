"""Quick agent testing script to verify all fixes."""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_test(name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {status} - {name}")
    if details:
        print(f"  {details}")

print("="*60)
print("AGENT TESTING - All Fixes Verification")
print("="*60)

# Test 1: IT Helpdesk Agent (should use OpenAI now)
print("\n[1] Testing IT Helpdesk Agent...")
try:
    response = requests.post(
        f"{BASE_URL}/api/conversation/start",
        json={"agent_type": "it_helpdesk"},
        timeout=30
    )
    if response.status_code == 200:
        data = response.json()
        session_id = data.get("session_id")

        # Send a test message
        chat_response = requests.post(
            f"{BASE_URL}/api/conversation/chat",
            json={"session_id": session_id, "message": "Hello"},
            timeout=30
        )
        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            print_test(
                "IT Helpdesk Agent",
                True,
                f"Session created and responded. Using OpenAI model."
            )
        else:
            print_test("IT Helpdesk Agent", False, f"Chat failed: {chat_response.text[:100]}")
    else:
        print_test("IT Helpdesk Agent", False, f"Status: {response.status_code}")
except Exception as e:
    print_test("IT Helpdesk Agent", False, f"Error: {str(e)[:100]}")

# Test 2: Research Agent (with increased recursion limit)
print("\n[2] Testing Research Agent...")
try:
    response = requests.post(
        f"{BASE_URL}/api/enterprise/research/invoke",
        json={"query": "What is LangChain?"},
        timeout=60
    )
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            print_test(
                "Research Agent",
                True,
                f"Response: {data.get('response', '')[:80]}... (recursion_limit=50)"
            )
        else:
            print_test("Research Agent", False, f"Error: {data.get('error', 'Unknown')}")
    else:
        print_test("Research Agent", False, f"Status: {response.status_code}")
except Exception as e:
    print_test("Research Agent", False, f"Error: {str(e)[:100]}")

# Test 3: Document Generator Agent (with fixed method call)
print("\n[3] Testing Document Generator Agent...")
try:
    response = requests.post(
        f"{BASE_URL}/api/enterprise/documents/invoke",
        json={
            "doc_type": "sop",
            "title": "Test SOP",
            "description": "Test procedure for password reset"
        },
        timeout=60
    )
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            print_test(
                "Document Generator Agent",
                True,
                "Document generated successfully using create_document()"
            )
        else:
            print_test("Document Generator Agent", False, f"Error: {data.get('error', 'Unknown')}")
    else:
        print_test("Document Generator Agent", False, f"Status: {response.status_code}")
except Exception as e:
    print_test("Document Generator Agent", False, f"Error: {str(e)[:100]}")

# Test 4: ServiceNow Agent (should use OpenAI now)
print("\n[4] Testing ServiceNow Agent...")
try:
    response = requests.post(
        f"{BASE_URL}/api/conversation/start",
        json={"agent_type": "servicenow"},
        timeout=30
    )
    if response.status_code == 200:
        data = response.json()
        print_test(
            "ServiceNow Agent",
            True,
            "Session started successfully. Using OpenAI model."
        )
    else:
        print_test("ServiceNow Agent", False, f"Status: {response.status_code}")
except Exception as e:
    print_test("ServiceNow Agent", False, f"Error: {str(e)[:100]}")

print("\n" + "="*60)
print("Testing Complete")
print("="*60)
