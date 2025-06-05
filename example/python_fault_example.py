from pyrgm import RGM2, add_faults
import numpy as np

if __name__ == "__main__":
    model = RGM2(n1=400, n2=420, nl=40, seed=324343)
    image = model.generate()
    image = add_faults(
        image,
        nf=17,
        dip_range=(60.0, 120.0),
        disp_range=(-10.0, 10.0),
        seed=324343,
    )
    image.astype("<f4").tofile("python_fault_example.bin")
    print("generated python_fault_example.bin")
