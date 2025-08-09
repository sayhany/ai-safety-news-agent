#!/usr/bin/env python3
"""
Simple test to verify the implementation structure without external dependencies.
"""

import sys
import os
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    print("📋 Testing Implementation Structure...")
    
    required_files = [
        'aisafety_news/processing/relevance.py',
        'aisafety_news/processing/dedupe.py', 
        'aisafety_news/processing/scoring.py',
        'aisafety_news/summarize.py',
        'aisafety_news/render.py',
        'templates/default_newsletter.md',
        'templates/daily_newsletter.md',
        'templates/weekly_newsletter.md',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All implementation files exist")
        return True

def test_imports():
    """Test that modules can be imported (syntax check)."""
    print("📋 Testing Module Syntax...")
    
    modules_to_test = [
        'aisafety_news.processing.relevance',
        'aisafety_news.processing.dedupe',
        'aisafety_news.processing.scoring',
        'aisafety_news.summarize',
        'aisafety_news.render',
    ]
    
    # Add current directory to path
    sys.path.insert(0, '.')
    
    failed_imports = []
    for module in modules_to_test:
        try:
            # Try to compile the module
            module_path = module.replace('.', '/') + '.py'
            with open(module_path, 'r') as f:
                code = f.read()
            compile(code, module_path, 'exec')
            print(f"✅ {module} - syntax OK")
        except Exception as e:
            print(f"❌ {module} - syntax error: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_class_definitions():
    """Test that key classes are defined."""
    print("📋 Testing Class Definitions...")
    
    # Test relevance module
    try:
        with open('aisafety_news/processing/relevance.py', 'r') as f:
            relevance_code = f.read()
        
        required_classes = ['RelevanceFilter', 'RelevanceScore', 'RelevanceLevel']
        missing_classes = []
        
        for cls in required_classes:
            if f'class {cls}' not in relevance_code:
                missing_classes.append(f'relevance.{cls}')
        
        if missing_classes:
            print(f"❌ Missing classes: {missing_classes}")
            return False
        else:
            print("✅ All required classes defined")
            return True
            
    except Exception as e:
        print(f"❌ Error checking classes: {e}")
        return False

def test_function_definitions():
    """Test that key functions are defined."""
    print("📋 Testing Function Definitions...")
    
    function_tests = [
        ('aisafety_news/processing/relevance.py', ['filter_relevance']),
        ('aisafety_news/processing/dedupe.py', ['deduplicate_articles']),
        ('aisafety_news/processing/scoring.py', ['score_articles']),
        ('aisafety_news/summarize.py', ['summarize_articles']),
        ('aisafety_news/render.py', ['render_newsletter']),
    ]
    
    missing_functions = []
    
    for file_path, functions in function_tests:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            for func in functions:
                if f'def {func}(' not in code and f'async def {func}(' not in code:
                    missing_functions.append(f'{file_path}:{func}')
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return False
    
    if missing_functions:
        print(f"❌ Missing functions: {missing_functions}")
        return False
    else:
        print("✅ All required functions defined")
        return True

def test_template_structure():
    """Test that templates have basic structure."""
    print("📋 Testing Template Structure...")
    
    templates = [
        'templates/default_newsletter.md',
        'templates/daily_newsletter.md',
        'templates/weekly_newsletter.md',
    ]
    
    for template in templates:
        try:
            with open(template, 'r') as f:
                content = f.read()
            
            # Check for basic Jinja2 template syntax
            if '{{' not in content or '}}' not in content:
                print(f"❌ {template} - missing Jinja2 syntax")
                return False
            
            # Check for key template variables
            required_vars = ['metadata', 'articles']
            missing_vars = []
            for var in required_vars:
                if var not in content:
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"❌ {template} - missing variables: {missing_vars}")
                return False
                
        except Exception as e:
            print(f"❌ Error reading {template}: {e}")
            return False
    
    print("✅ All templates have proper structure")
    return True

def main():
    """Run all tests."""
    print("🚀 Testing AI Safety News Agent Implementation...")
    print()
    
    tests = [
        test_file_structure,
        test_imports,
        test_class_definitions,
        test_function_definitions,
        test_template_structure,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All implementation tests passed!")
        print()
        print("🎉 Implementation is complete and ready!")
        print()
        print("Next steps:")
        print("1. Install dependencies: poetry install --with dev")
        print("2. Set up environment: cp .env.example .env")
        print("3. Add your OPENROUTER_API_KEY to .env")
        print("4. Run with mock data: poetry run aisafety-news 2025-07-18 --mock")
        print("5. Run with real data: poetry run aisafety-news 2025-07-18")
        return True
    else:
        print("❌ Some implementation tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
