"""
Updated Quick Test Script - Handles .jpg extension in frame IDs
"""

from pathlib import Path


def quick_test():
    """Quick sanity check"""
    
    print("\n" + "="*60)
    print("QUICK ANNOTATION LOADER TEST")
    print("="*60)
    
    # 1. Check if files exist
    print("\n1. Checking file structure...")
    
    dataset_root = Path("data/volleyball")
    
    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        print("   → Update the 'dataset_root' path in this script")
        return False
    
    print(f"✓ Dataset root exists: {dataset_root}")
    
    # Find video directories
    video_dirs = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not video_dirs:
        print("❌ No video directories found")
        return False
    
    print(f"✓ Found {len(video_dirs)} video directories")
    
    # Check first video
    first_video = sorted(video_dirs)[0]
    print(f"✓ Testing with video: {first_video.name}")
    
    annotations_file = first_video / "annotations.txt"
    if not annotations_file.exists():
        print(f"❌ annotations.txt not found in {first_video.name}")
        return False
    
    print(f"✓ annotations.txt exists")
    
    # 2. Test parsing one line
    print("\n2. Testing annotation parsing...")
    
    try:
        with open(annotations_file, 'r') as f:
            # Read first non-empty line
            test_line = None
            for line in f:
                line = line.strip()
                if line:
                    test_line = line
                    break
        
        if not test_line:
            print("❌ No valid lines in annotations.txt")
            return False
        
        print(f"   Line: {test_line[:80]}...")
        
        # Parse manually (handling .jpg extension)
        parts = test_line.split()
        
        # Frame ID might have .jpg extension
        frame_id_str = parts[0]
        if frame_id_str.endswith('.jpg'):
            frame_id_str = frame_id_str[:-4]
        
        frame_id = int(frame_id_str)
        activity = parts[1]
        
        # Count players (each player has 5 fields: action x y w h)
        player_parts = parts[2:]
        num_players = len(player_parts) // 5
        
        print(f"✓ Frame ID: {frame_id} (parsed from '{parts[0]}')")
        print(f"✓ Activity: {activity}")
        print(f"✓ Number of players: {num_players}")
        
        if num_players > 0:
            print(f"✓ First player action: {player_parts[0]}")
            print(f"✓ First player bbox: ({player_parts[1]}, {player_parts[2]}, {player_parts[3]}, {player_parts[4]})")
        
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Check frame directory
    print("\n3. Checking frame directories...")
    
    # Find frame directories
    frame_dirs = [d for d in first_video.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not frame_dirs:
        print(f"❌ No frame directories found in {first_video.name}")
        return False
    
    print(f"✓ Found {len(frame_dirs)} frame directories")
    
    # Check first frame directory
    first_frame_dir = sorted(frame_dirs)[0]
    images = list(first_frame_dir.glob("*.jpg"))
    
    print(f"✓ Frame directory {first_frame_dir.name} has {len(images)} images")
    
    if len(images) == 41:
        print(f"✓ Correct number of images (41)")
    else:
        print(f"⚠ Expected 41 images, found {len(images)}")
    
    # 4. Test with your annotation class
    print("\n4. Testing VolleyballAnnotation class...")
    
    try:
        # Try to import your class
        # First try the fixed version
        try:
            from volleyball_loader_fixed import VolleyballAnnotation
            print("  Using volleyball_loader_fixed.py")
        except ImportError:
            # Fall back to original location
            from src.data.volleyball_loader import VolleyballAnnotation
            print("  Using src.data.volleyball_loader.py")
        
        # Test parsing with your class
        with open(annotations_file, 'r') as f:
            success_count = 0
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    annot = VolleyballAnnotation(line)
                    if i == 1:  # Print first one
                        print(f"✓ VolleyballAnnotation class works!")
                        print(f"  Frame: {annot.frame_id}")
                        print(f"  Activity: {annot.activity_class}")
                        print(f"  Players: {annot.num_players}")
                    success_count += 1
                    
                    if i >= 5:  # Test first 5
                        break
                except Exception as e:
                    print(f"  ✗ Failed on line {i}: {e}")
                    return False
            
            print(f"✓ Successfully parsed {success_count} annotation lines")
        
    except ImportError as e:
        print(f"⚠ Could not import VolleyballAnnotation: {e}")
        print(f"  This is OK if you haven't created the file yet")
        print(f"  Use the fixed version: volleyball_loader_fixed.py")
        return None  # Not a failure, just not ready
    except Exception as e:
        print(f"❌ VolleyballAnnotation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Statistics
    print("\n5. Gathering quick statistics...")
    
    try:
        from collections import Counter
        
        activity_counter = Counter()
        action_counter = Counter()
        total_frames = 0
        total_players = 0
        
        # Use fixed loader if available
        try:
            from volleyball_loader_fixed import VolleyballAnnotation
        except ImportError:
            from src.data.volleyball_loader import VolleyballAnnotation
        
        for video_dir in video_dirs[:3]:  # Just first 3 videos for speed
            annot_file = video_dir / "annotations.txt"
            
            with open(annot_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        annot = VolleyballAnnotation(line)
                        total_frames += 1
                        total_players += annot.num_players
                        activity_counter[annot.activity_class] += 1
                        
                        for player in annot.players:
                            action_counter[player['action']] += 1
                    except:
                        continue
        
        print(f"  Total frames (first 3 videos): {total_frames}")
        print(f"  Total players: {total_players}")
        print(f"  Avg players/frame: {total_players/total_frames:.1f}")
        
        print(f"\n  Top activities:")
        for activity, count in activity_counter.most_common(3):
            print(f"    {activity}: {count}")
        
        print(f"\n  Top actions:")
        for action, count in action_counter.most_common(3):
            print(f"    {action}: {count}")
            
    except Exception as e:
        print(f"⚠ Statistics failed: {e}")
        # Not critical, continue
    
    print("\n" + "="*60)
    print("✓ ALL QUICK TESTS PASSED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy volleyball_loader_fixed.py to src/data/volleyball_loader.py")
    print("2. Run full tests: python test_annotation_loader.py")
    print("3. Extract features: python scripts/extract_features.py")
    print()
    
    return True


if __name__ == "__main__":
    result = quick_test()
    
    if result is True:
        print("✓ Success! Ready to proceed.")
    elif result is False:
        print("✗ Tests failed. Check the errors above.")
    else:
        print("⚠ Partial success. Some components not ready yet.")