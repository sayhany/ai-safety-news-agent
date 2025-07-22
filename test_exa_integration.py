#!/usr/bin/env python3
"""
Direct test of Exa Search API integration without pytest.
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from aisafety_news.config import get_settings, get_model_config
from aisafety_news.ingest.sources import SearchAPIAdapter, ExaSearchAPIAdapter


async def test_api_key_loading():
    """Test 1: Verify API key is loaded correctly."""
    print("=" * 60)
    print("TEST 1: API Key Loading")
    print("=" * 60)
    
    settings = get_settings()
    
    if settings.exa_api_key:
        print("✅ EXA_API_KEY loaded successfully")
        print(f"   Key starts with: {settings.exa_api_key[:8]}...")
        return True
    else:
        print("❌ EXA_API_KEY not loaded")
        print(f"   Value: {settings.exa_api_key}")
        return False


async def test_search_source_config():
    """Test 2: Verify search sources are configured correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: Search Source Configuration")
    print("=" * 60)
    
    try:
        model_config = get_model_config()
        search_sources = model_config.get_search_sources()
        
        if not search_sources:
            print("❌ No search sources configured")
            return False
        
        print(f"✅ Found {len(search_sources)} search sources")
        
        exa_sources = [s for s in search_sources if s.api == 'exa']
        print(f"   Exa sources: {len(exa_sources)}")
        
        # Show first few sources
        for i, source in enumerate(exa_sources[:3]):
            print(f"   {i+1}. {source.name}")
            print(f"      Query: {source.query[:50]}...")
            print(f"      Priority: {source.priority}")
        
        return len(exa_sources) > 0
        
    except Exception as e:
        print(f"❌ Error loading search sources: {e}")
        return False


async def test_adapter_instantiation():
    """Test 3: Test SearchAPIAdapter instantiation."""
    print("\n" + "=" * 60)
    print("TEST 3: Adapter Instantiation")
    print("=" * 60)
    
    try:
        model_config = get_model_config()
        search_sources = model_config.get_search_sources()
        
        if not search_sources:
            print("❌ No search sources to test")
            return False
        
        # Test with first Exa source
        test_source = search_sources[0]
        print(f"Testing with source: {test_source.name}")
        
        adapter = SearchAPIAdapter(test_source)
        print(f"✅ SearchAPIAdapter created successfully")
        print(f"   API: {adapter.api}")
        print(f"   Query: {adapter.query[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating adapter: {e}")
        return False


async def test_exa_adapter_direct():
    """Test 4: Test ExaSearchAPIAdapter directly."""
    print("\n" + "=" * 60)
    print("TEST 4: Direct Exa Adapter Test")
    print("=" * 60)
    
    try:
        model_config = get_model_config()
        search_sources = model_config.get_search_sources()
        
        if not search_sources:
            print("❌ No search sources to test")
            return False
        
        test_source = search_sources[0]
        print(f"Testing with source: {test_source.name}")
        
        adapter = ExaSearchAPIAdapter(test_source)
        print(f"✅ ExaSearchAPIAdapter created successfully")
        
        # Test API key validation
        settings = get_settings()
        if not settings.exa_api_key:
            print("❌ No API key available for testing")
            return False
        
        print("✅ API key validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error with ExaSearchAPIAdapter: {e}")
        return False


async def test_real_api_call():
    """Test 5: Make a real API call to Exa (if API key is valid)."""
    print("\n" + "=" * 60)
    print("TEST 5: Real API Call Test")
    print("=" * 60)
    
    try:
        settings = get_settings()
        if not settings.exa_api_key:
            print("❌ No API key available for real API test")
            return False
        
        model_config = get_model_config()
        search_sources = model_config.get_search_sources()
        
        if not search_sources:
            print("❌ No search sources configured")
            return False
        
        # Use a simple, reliable source for testing
        test_source = None
        for source in search_sources:
            if "Reuters" in source.name:
                test_source = source
                break
        
        if not test_source:
            test_source = search_sources[0]
        
        print(f"Testing API call with: {test_source.name}")
        print(f"Query: {test_source.query}")
        
        adapter = SearchAPIAdapter(test_source)
        
        # Test with a recent date range
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        
        async with adapter:
            print("🔄 Making API call...")
            articles = await adapter.fetch_articles(start_date, limit=3)
            
            if articles:
                print(f"✅ API call successful! Retrieved {len(articles)} articles")
                for i, article in enumerate(articles[:2]):
                    print(f"   {i+1}. {article['title'][:60]}...")
                    print(f"      URL: {article['url']}")
                    print(f"      Published: {article['published_date']}")
                return True
            else:
                print("⚠️  API call successful but no articles returned")
                return True
                
    except Exception as e:
        print(f"❌ Real API call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


async def main():
    """Run all tests."""
    print("🔍 TESTING EXA SEARCH API INTEGRATION")
    print("=" * 60)
    
    tests = [
        test_api_key_loading,
        test_search_source_config,
        test_adapter_instantiation,
        test_exa_adapter_direct,
        test_real_api_call,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "API Key Loading",
        "Search Source Configuration", 
        "Adapter Instantiation",
        "Direct Exa Adapter Test",
        "Real API Call Test"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed - see details above")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)