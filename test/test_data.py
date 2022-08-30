from diffdrr import load_example_ct


def test_load_example_ct():
    volume, spacing = load_example_ct()
    assert volume.shape == (512, 512, 133)
    assert spacing == [0.703125, 0.703125, 2.5]
