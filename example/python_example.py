from pyrgm.rgm2 import RGM2
import numpy as np

if __name__ == "__main__":
    model = RGM2(n1=256, n2=256, seed=2024)
    image = model.generate()
    image.astype('<f4').tofile('python_example_dhr.bin')
    print("generated python_example_dhr.bin")
