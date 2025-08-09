#!/usr/bin/env python3
"""Test OpenRouter API integration."""

import os
import asyncio
from dotenv import load_dotenv
import openai

async def test_openrouter():
    """Test OpenRouter API connection."""
    print("🔍 TESTING OPENROUTER API INTEGRATION")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print(f"API Key loaded: {'✅' if api_key else '❌'}")
    if api_key:
        print(f"   Key starts with: {api_key[:20]}...")
    
    if not api_key:
        print("❌ No API key found")
        return
    
    # Test API call
    try:
        client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        print("\n🔄 Testing API call...")
        response = await client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "user", "content": "Say 'API test successful' if you can read this."}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"✅ API call successful!")
        print(f"   Response: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_openrouter())