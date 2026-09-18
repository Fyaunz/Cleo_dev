# Run this INSIDE Blender with your animated object selected!
import bpy # type: ignore
import json
import math

def bake_animation_100hz(filepath):
    obj = bpy.context.active_object
    if not obj:
        print("No active object selected!")
        return

    scene = bpy.context.scene
    original_frame = scene.frame_current
    
    # Force evaluation at 100 frames per second
    target_fps = 100
    start_frame = scene.frame_start
    end_frame = scene.frame_end
    
    stream_data = []
    
    print(f"Baking animation from frame {start_frame} to {end_frame} at 100Hz...")

    # Iterate through every single frame on the timeline
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame) # Physically advance Blender's timeline
        
        # Calculate timestamps based on a 50Hz clock
        time_sec = (frame - start_frame) / target_fps
        
        # Read the exact world-space Euler rotations (handles modifiers/constraints too)
        matrix = obj.matrix_world.to_euler('XYZ')
        
        pitch = math.degrees(matrix.x)
        roll = math.degrees(matrix.y)
        yaw = math.degrees(matrix.z)
        
        stream_data.append({
            "time_sec": round(time_sec, 3),
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2)
        })

    # Restore the timeline playhead location
    scene.frame_set(original_frame)

    with open(filepath, 'w') as f:
        json.dump(stream_data, f, indent=2)
        
    print(f"Successfully baked {len(stream_data)} frames to {filepath}")

# Update to your local project path
bake_animation_100hz("/Users/your_username/cleodev/demo/baked_animation.json")