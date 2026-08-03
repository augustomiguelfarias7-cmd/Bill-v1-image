# Bill v1 - Image Model

Experimental lightweight U-Net image model.

## Overview

Bill v1 is a small U-Net based model trained for educational and experimental purposes.  
It performs basic image reconstruction and generation at low resolution (64x64).

**Model details:**
- Architecture: U-Net
- Parameters: ~379,000
- Resolution: 64×64
- Framework: PyTorch

**Main weights file:** `bill_unet_full.pth` (located in the root of the repository)

## How to run

```bash
pip install -r requirements.txt
python inference.py
```

The script loads the weights from `bill_unet_full.pth` and generates an example image.

## License

This project is released under the **General Permissive of Language Models** license.  
See the `LICENSE` file for full terms.

## Disclaimer

This is an experimental educational model. Output quality is limited.
