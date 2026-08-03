"""
Script to reconstruct the full weights file from the split parts.
"""
import os

def join_weights(parts_dir="weights/parts", output="weights/bill_unet.pth"):
    parts = sorted([f for f in os.listdir(parts_dir) if f.endswith(".bin")])
    
    if not parts:
        print("No part files found.")
        return
    
    print(f"Found {len(parts)} parts. Joining...")
    
    with open(output, "wb") as out:
        for part in parts:
            path = os.path.join(parts_dir, part)
            with open(path, "rb") as f:
                data = f.read()
                out.write(data)
            print(f"  + {part} ({len(data)/1024:.1f} KB)")
    
    size = os.path.getsize(output)
    print(f"\nDone! Full weights saved to: {output}")
    print(f"Total size: {size/1024:.1f} KB")

if __name__ == "__main__":
    join_weights()
