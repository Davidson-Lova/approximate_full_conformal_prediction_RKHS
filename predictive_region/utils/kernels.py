from sklearn.metrics.pairwise import laplacian_kernel, polynomial_kernel


class kernels:
    def __init__(self):
        pass

    def maker(self, name):
        if name == "polynomial":
            return polynomial_kernel
        elif name == "laplacian":
            return laplacian_kernel
        else:
            print("Not listed")
            return polynomial_kernel
