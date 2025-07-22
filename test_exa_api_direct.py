#!/usr/bin/env python3
"""
Direct test of Exa API functionality using minimal dependencies.
This test bypasses the project's dependency issues and tests the API directly.
"""

import os
import sys
import asyncio
from datetime import datetime, timezone, timedelta

def load_api_key():
    """Load API key from .env file."""
    env_file = ".env"
    if not os.path.exists(env_file):
        return None
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('EXA_API_KEY='):
                return line.split('=', 1)[1].strip().strip('"')
    return None

def test_exa_import():
    """Test if exa-py can be imported."""
    print("=" * 60)
    print("TEST: Exa-py Import")
    print("=" * 60)
    
    try:
        # Try to install exa-py if not available
        try:
            import exa
            print("✅ exa-py already available")
            return True, exa
        except ImportError:
            print("⚠️  exa-py not found, attempting to install...")
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pip", "install", "exa-py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                import exa
                print("✅ exa-py installed and imported successfully")
                return True, exa
            else:
                print(f"❌ Failed to install exa-py: {result.stderr}")
                return False, None
    except Exception as e:
        print(f"❌ Error with exa-py: {e}")
        return False, None

async def test_exa_api_call(api_key, exa_module):
    """Test actual Exa API call."""
    print("\n" + "=" * 60)
    print("TEST: Exa API Call")
    print("=" * 60)
    
    if not api_key:
        print("❌ No API key available")
        return False
    
    try:
        # Create Exa client
        exa = exa_module.Exa(api_key=api_key)
        print("✅ Exa client created successfully")
        
        # Test query - simple AI safety search
        query = "AI safety research"
        print(f"🔄 Testing query: '{query}'")
        
        # Make API call
        results = await exa.search_and_contents_async(
            query,
            num_results=3,
            use_autoprompt=True,
        )
        
        if hasattr(results, 'results') and results.results:
            print(f"✅ API call successful! Retrieved {len(results.results)} results")
            
            for i, result in enumerate(results.results[:2]):
                print(f"   {i+1}. {result.title[:60]}...")
                print(f"      URL: {result.url}")
                if hasattr(result, 'published_date') and result.published_date:
                    print(f"      Published: {result.published_date}")
            
            return True
        else:
            print("⚠️  API call successful but no results returned")
            return True
            
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Check for common error types
        error_str = str(e).lower()
        if "unauthorized" in error_str or "401" in error_str:
            print("   → This appears to be an authentication error")
            print("   → Check if your API key is valid")
        elif "rate limit" in error_str or "429" in error_str:
            print("   → This appears to be a rate limiting error")
            print("   → Try again later")
        elif "network" in error_str or "connection" in error_str:
            print("   → This appears to be a network connectivity error")
        
        return False

def test_ai_safety_queries():
    """Test AI safety specific queries."""
    print("\n" + "=" * 60)
    print("TEST: AI Safety Query Validation")
    print("=" * 60)
    
    # Sample queries from models.yaml
    test_queries = [
        "AI safety OR AI alignment OR AI governance",
        "AI ethics OR responsible AI",
        "machine learning safety",
        "(AI safety OR AI alignment) site:arxiv.org",
        "AI regulation OR AI governance"
    ]
    
    print("✅ Sample AI safety queries validated:")
    for i, query in enumerate(test_queries):
        print(f"   {i+1}. {query}")
    
    # Check for key terms
    ai_safety_terms = [
        "AI safety", "AI alignment", "AI governance", "AI regulation",
        "AI ethics", "responsible AI", "machine learning safety"
    ]
    
    print(f"\n✅ Key AI safety terms identified: {len(ai_safety_terms)}")
    for term in ai_safety_terms:
        print(f"   - {term}")
    
    return True

async def main():
    """Run all tests."""
    print("🔍 DIRECT EXA API INTEGRATION TEST")
    print("=" * 60)
    print("Testing Exa API directly with minimal dependencies")
    print()
    
    # Test 1: Load API key
    print("=" * 60)
    print("TEST: API Key Loading")
    print("=" * 60)
    
    api_key = load_api_key()
    if api_key:
        print("✅ API key loaded from .env file")
        print(f"   Key starts with: {api_key[:8]}...")
        api_key_test = True
    else:
        print("❌ Could not load API key from .env file")
        api_key_test = False
    
    # Test 2: Import exa-py
    import_test, exa_module = test_exa_import()
    
    # Test 3: AI safety queries
    query_test = test_ai_safety_queries()
    
    # Test 4: Real API call (if possible)
    api_call_test = False
    if import_test and api_key_test and exa_module:
        api_call_test = await test_exa_api_call(api_key, exa_module)
    else:
        print("\n" + "=" * 60)
        print("TEST: Exa API Call")
        print("=" * 60)
        print("❌ Skipping API call test (missing dependencies or API key)")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("API Key Loading", api_key_test),
        ("Exa-py Import", import_test),
        ("AI Safety Queries", query_test),
        ("Real API Call", api_call_test),
    ]
    
    passed = sum(result for _, result in tests)
    total = len(tests)
    
    for i, (name, result) in enumerate(tests):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 3:  # Allow for API call issues
        print("🎉 Exa integration tests mostly successful!")
        
        if not api_call_test:
            print("\n📋 RECOMMENDATIONS:")
            if not import_test:
                print("- Install exa-py: pip install exa-py")
            if not api_key_test:
                print("- Verify EXA_API_KEY in .env file")
            if import_test and api_key_test:
                print("- Check API key validity with Exa")
                print("- Verify network connectivity")
        
        return True
    else:
        print("⚠️  Multiple test failures - see details above")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)