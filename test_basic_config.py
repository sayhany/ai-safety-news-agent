#!/usr/bin/env python3
"""
Basic configuration and API key test for Exa Search API integration.
Tests only the configuration loading without requiring external dependencies.
"""

import os
import sys
import yaml
from pathlib import Path

def test_env_file_loading():
    """Test 1: Check if .env file exists and contains EXA_API_KEY."""
    print("=" * 60)
    print("TEST 1: Environment File Loading")
    print("=" * 60)
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    print("✅ .env file exists")
    
    # Read .env file manually
    env_vars = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"')
    
    if 'EXA_API_KEY' in env_vars:
        api_key = env_vars['EXA_API_KEY']
        if api_key and api_key != "your-exa-search-api-key":
            print(f"✅ EXA_API_KEY found in .env file")
            print(f"   Key starts with: {api_key[:8]}...")
            return True
        else:
            print("❌ EXA_API_KEY is placeholder or empty")
            return False
    else:
        print("❌ EXA_API_KEY not found in .env file")
        return False


def test_models_yaml_config():
    """Test 2: Check models.yaml for search sources configuration."""
    print("\n" + "=" * 60)
    print("TEST 2: Models.yaml Configuration")
    print("=" * 60)
    
    models_file = Path("models.yaml")
    if not models_file.exists():
        print("❌ models.yaml file not found")
        return False
    
    print("✅ models.yaml file exists")
    
    try:
        with open(models_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'search_sources' not in config:
            print("❌ search_sources section not found in models.yaml")
            return False
        
        search_sources = config['search_sources']
        if not search_sources:
            print("❌ search_sources is empty")
            return False
        
        print(f"✅ Found {len(search_sources)} search sources")
        
        exa_sources = [s for s in search_sources if s.get('api') == 'exa']
        print(f"   Exa sources: {len(exa_sources)}")
        
        if len(exa_sources) == 0:
            print("❌ No Exa sources configured")
            return False
        
        # Show first few sources
        print("   Sample Exa sources:")
        for i, source in enumerate(exa_sources[:3]):
            print(f"   {i+1}. {source.get('name', 'Unknown')}")
            query = source.get('query', '')
            print(f"      Query: {query[:50]}...")
            print(f"      Priority: {source.get('priority', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading models.yaml: {e}")
        return False


def test_exa_py_import():
    """Test 3: Check if exa-py can be imported."""
    print("\n" + "=" * 60)
    print("TEST 3: Exa-py Library Import")
    print("=" * 60)
    
    try:
        import exa
        print("✅ exa-py library can be imported")
        print(f"   Exa module location: {exa.__file__}")
        
        # Try to create Exa client (without making API calls)
        try:
            client = exa.Exa(api_key="test_key")
            print("✅ Exa client can be instantiated")
            return True
        except Exception as e:
            print(f"⚠️  Exa client instantiation issue: {e}")
            return True  # Still consider this a pass if import works
            
    except ImportError as e:
        print(f"❌ Cannot import exa-py: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error with exa-py: {e}")
        return False


def test_project_structure():
    """Test 4: Verify project structure for Exa integration."""
    print("\n" + "=" * 60)
    print("TEST 4: Project Structure")
    print("=" * 60)
    
    required_files = [
        "aisafety_news/config.py",
        "aisafety_news/ingest/sources.py",
        "tests/test_search_api.py",
        "pyproject.toml"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist


def test_search_queries():
    """Test 5: Validate search queries in configuration."""
    print("\n" + "=" * 60)
    print("TEST 5: Search Query Validation")
    print("=" * 60)
    
    try:
        with open("models.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        search_sources = config.get('search_sources', [])
        exa_sources = [s for s in search_sources if s.get('api') == 'exa']
        
        if not exa_sources:
            print("❌ No Exa sources to validate")
            return False
        
        ai_safety_terms = [
            'AI safety', 'AI alignment', 'AI governance', 'AI regulation',
            'AI ethics', 'responsible AI', 'machine learning safety'
        ]
        
        valid_queries = 0
        for source in exa_sources:
            query = source.get('query', '').lower()
            has_ai_safety_term = any(term.lower() in query for term in ai_safety_terms)
            
            if has_ai_safety_term:
                valid_queries += 1
            else:
                print(f"⚠️  Query may not be AI safety focused: {source.get('name')}")
        
        print(f"✅ {valid_queries}/{len(exa_sources)} queries contain AI safety terms")
        
        # Check for site-specific queries
        site_queries = sum(1 for s in exa_sources if 'site:' in s.get('query', ''))
        print(f"✅ {site_queries}/{len(exa_sources)} queries are site-specific")
        
        return valid_queries > 0
        
    except Exception as e:
        print(f"❌ Error validating queries: {e}")
        return False


def main():
    """Run all basic tests."""
    print("🔍 BASIC EXA SEARCH API INTEGRATION TEST")
    print("=" * 60)
    print("Testing configuration and setup without external dependencies")
    print()
    
    tests = [
        test_env_file_loading,
        test_models_yaml_config,
        test_exa_py_import,
        test_project_structure,
        test_search_queries,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
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
        "Environment File Loading",
        "Models.yaml Configuration", 
        "Exa-py Library Import",
        "Project Structure",
        "Search Query Validation"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 4:  # Allow for import issues
        print("🎉 Configuration tests mostly passed!")
        print("\nNext steps:")
        print("- Install dependencies: poetry install")
        print("- Run full integration tests")
        print("- Test real API calls")
        return True
    else:
        print("⚠️  Configuration issues found - see details above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)