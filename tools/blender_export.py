# Run this INSIDE Blender!
import bpy # type: ignore
import json
import math

def export_keyframes(filepath):
    obj = bpy.context.active_object
    if not obj or not obj.animation_data or not obj.animation_data.action:
        print("No active animated object found!")
        return

    fps = bpy.context.scene.render.fps
    action = obj.animation_data.action
    
    # We will store the timeline here
    # Format: { frame_number: {"time_sec": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0} }
    timeline = {}

    for fcurve in action.fcurves:
        # data_path 'rotation_euler' array index: 0=X(Pitch), 1=Y(Roll), 2=Z(Yaw)
        if fcurve.data_path == "rotation_euler":
            axis = fcurve.array_index
            
            for keyframe in fcurve.keyframe_points:
                frame = int(keyframe.co.x)
                value_radians = keyframe.co.y
                value_degrees = math.degrees(value_radians)
                
                if frame not in timeline:
                    timeline[frame] = {
                        "time_sec": frame / fps,
                        "yaw": 0.0, "pitch": 0.0, "roll": 0.0
                    }
                
                if axis == 0: timeline[frame]["pitch"] = value_degrees
                if axis == 1: timeline[frame]["roll"] = value_degrees
                if axis == 2: timeline[frame]["yaw"] = value_degrees

    # Sort the dictionary by frame number to ensure chronological playback
    sorted_frames = [timeline[k] for k in sorted(timeline.keys())]

    with open(filepath, 'w') as f:
        json.dump(sorted_frames, f, indent=4)
        
    print(f"Exported {len(sorted_frames)} keyframes to {filepath}")

# Change this path to where your Python SDK lives!
export_keyframes("/Users/your_username/cleodev/demo/animation.json")