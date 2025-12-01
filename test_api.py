#!/usr/bin/env python3
"""
Test script for IndexTTS API endpoints
"""

import requests
import json
import time
import os

# API base URL
BASE_URL = "http://localhost:7871"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_system_info():
    """Test system info endpoint"""
    print("\n🔍 Testing system info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/system/info")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ System info retrieved:")
            print(f"   GPU Available: {info['system']['gpu_available']}")
            print(f"   Model Version: {info['model']['version']}")
            print(f"   Demo Voices: {info['capabilities']['demo_voices']}")
            return True
        else:
            print(f"❌ System info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ System info failed: {e}")
        return False

def test_demo_categories():
    """Test demo categories endpoint"""
    print("\n🔍 Testing demo categories endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/demo/categories")
        if response.status_code == 200:
            data = response.json()
            categories = data.get('categories', [])
            print(f"✅ Found {len(categories)} demo categories:")
            for category in categories[:3]:  # Show first 3
                print(f"   - {category['name']}: {len(category['subcategories'])} subcategories")
            return categories
        else:
            print(f"❌ Demo categories failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Demo categories failed: {e}")
        return []

def test_demo_voices(category, subcategory):
    """Test demo voices endpoint"""
    print(f"\n🔍 Testing demo voices for {category}/{subcategory}...")
    try:
        response = requests.get(f"{BASE_URL}/api/demo/voices/{category}/{subcategory}")
        if response.status_code == 200:
            data = response.json()
            voices = data.get('voices', [])
            print(f"✅ Found {len(voices)} voices:")
            for voice in voices[:3]:  # Show first 3
                print(f"   - {voice['filename']}")
            return voices
        else:
            print(f"❌ Demo voices failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Demo voices failed: {e}")
        return []

def test_demo_voice_tts(category, subcategory, filename):
    """Test demo voice TTS generation"""
    print(f"\n🔍 Testing demo voice TTS with {filename}...")
    try:
        payload = {
            "category": category,
            "subcategory": subcategory,
            "filename": filename,
            "text": "你好，这是一个API测试",
            "parameters": {
                "temperature": 0.8,
                "top_p": 0.9
            }
        }
        
        response = requests.post(f"{BASE_URL}/api/demo/use", json=payload)
        if response.status_code == 200:
            data = response.json()
            task_id = data.get('task_id')
            print(f"✅ Demo TTS started, task ID: {task_id}")
            
            # Wait for completion
            return wait_for_task(task_id)
        else:
            print(f"❌ Demo TTS failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Demo TTS failed: {e}")
        return False

def test_queue_status():
    """Test queue status endpoint"""
    print("\n🔍 Testing queue status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/queue/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Queue status:")
            print(f"   Queue size: {data.get('queue_size', 0)}")
            print(f"   Total completed: {data.get('total_completed', 0)}")
            current = data.get('current_task')
            if current:
                print(f"   Current task: {current['id']}")
            return True
        else:
            print(f"❌ Queue status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Queue status failed: {e}")
        return False

def wait_for_task(task_id, timeout=60):
    """Wait for task completion"""
    print(f"⏳ Waiting for task {task_id} to complete...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/api/tts/status/{task_id}")
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                progress = data.get('progress', 0)
                
                if status == 'completed':
                    print(f"✅ Task {task_id} completed!")
                    # Try to get result
                    result_response = requests.get(f"{BASE_URL}/api/tts/result/{task_id}")
                    if result_response.status_code == 200:
                        print(f"✅ Audio file generated successfully")
                        return True
                    else:
                        print(f"❌ Could not retrieve result: {result_response.status_code}")
                        return False
                        
                elif status == 'failed':
                    error = data.get('error', 'Unknown error')
                    print(f"❌ Task {task_id} failed: {error}")
                    return False
                    
                elif status == 'running':
                    print(f"⏳ Task {task_id} running... ({progress*100:.1f}%)")
                    
                elif status == 'queued':
                    print(f"⏳ Task {task_id} queued...")
                    
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error checking task status: {e}")
            return False
    
    print(f"❌ Task {task_id} timed out")
    return False

def test_recent_audio():
    """Test recent audio endpoint"""
    print("\n🔍 Testing recent audio endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/audio/recent")
        if response.status_code == 200:
            data = response.json()
            files = data.get('files', [])
            print(f"✅ Found {len(files)} recent audio files")
            for file in files[:3]:  # Show first 3
                print(f"   - {file['filename']} ({file['size']} bytes)")
            return True
        else:
            print(f"❌ Recent audio failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Recent audio failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 IndexTTS API Test Suite")
    print("=" * 50)
    
    # Test basic endpoints
    if not test_health():
        print("❌ Server is not running. Please start the API server first.")
        return
    
    test_system_info()
    
    # Test demo functionality
    categories = test_demo_categories()
    if categories:
        # Test first available category/subcategory
        category = categories[0]
        if category['subcategories']:
            subcategory = category['subcategories'][0]
            voices = test_demo_voices(category['name'], subcategory['name'])
            if voices:
                # Test TTS with first voice
                test_demo_voice_tts(category['name'], subcategory['name'], voices[0]['filename'])
    
    # Test queue and audio endpoints
    test_queue_status()
    test_recent_audio()
    
    print("\n✅ API tests completed!")
    print(f"📖 Full documentation available at: {BASE_URL}/docs")

if __name__ == "__main__":
    main()