from torchvision.transforms import Resize, ToTensor
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.resizor = Resize(self.output_size)
        self.toter = ToTensor()
    def __call__(self, sample):
        new_image = self.toter(self.resizor(sample))
        return new_image

