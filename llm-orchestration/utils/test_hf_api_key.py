#!/usr/bin/env python3
"""Test HuggingFace API key validity"""

import httpx
import asyncio
import os
from dotenv import load_dotenv

async def test_hf_api():
    load_dotenv()
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    
    print(f"🔑 Testing API Key: {api_key[:10]}...{api_key[-5:]}")
    print("=" * 70)
    
    # Test with a simple classification request
    url = "https://router.huggingface.co/hf-inference/models/MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": "I love Python",
        "parameters": {
            "candidate_labels": ["opinion", "fact", "preference"]
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            print("📡 Sending test request...")
            response = await client.post(url, headers=headers, json=payload)
            
            print(f"\n📊 Response Status: {response.status_code}")
            print(f"📊 Response Headers:")
            for key, value in response.headers.items():
                if key.lower() in ['content-type', 'x-error-message', 'x-error-type']:
                    print(f"   {key}: {value}")
            
            print(f"\n📊 Response Body:")
            print(response.text[:500])
            
            if response.status_code == 200:
                print("\n✅ API Key is VALID and working!")
                result = response.json()
                print(f"\n📋 Classification Result:")
                for item in result:
                    print(f"   {item['label']}: {item['score']:.4f}")
            elif response.status_code == 401:
                print("\n❌ API Key is INVALID or EXPIRED!")
                print("   Please generate a new key at: https://huggingface.co/settings/tokens")
            elif response.status_code == 403:
                print("\n❌ API Key doesn't have permission for this model!")
            elif response.status_code == 503:
                print("\n⚠️  Model is loading (cold start) - this is normal, retry in a few seconds")
            else:
                print(f"\n⚠️  Unexpected status code: {response.status_code}")
                
        except httpx.TimeoutException:
            print("\n⏱️  Request timed out - HuggingFace API might be slow")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_hf_api())

