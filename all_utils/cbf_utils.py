import numpy as np
from sympy import Matrix, lambdify, simplify, symbols

class SphereProjectedCBF(object):
    """This class builds a CBF for a sphere projected on a plane. The perspective projection of a 
        sphere on a plane is an ellipse. The CBF is built based on the following derivation:
        https://math.stackexchange.com/questions/1367710/perspective-projection-of-a-sphere-on-a-plane
    """

    def __init__(self, r):   
        """Initialize the CBF for a sphere projected on a plane.

        Args:
            f (float): focal length
            r (float): radius of the sphere
        """
        self.r = r
        x, y, x0, y0, Zc = symbols('x y x0 y0 Zc')
        Q = Matrix([[y0**2 + 1 - self.r**2/Zc**2, -x0*y0],
                    [-x0*y0, x0**2 + 1 - self.r**2/Zc**2]])
        q = Matrix([-2*x0, -2*y0])
        c = Matrix([[x0**2 + y0**2 - self.r**2/Zc**2]])
        p = Matrix([x,y])
        h = simplify(p.T @ Q @ p + q.T @ p + c)
        self.h = lambdify([x, y, x0, y0, Zc], h, "numpy")
        partial_h = simplify(h.jacobian(Matrix([x, y, x0, y0, Zc])))
        self.partial_h = lambdify([x, y, x0, y0, Zc], partial_h, "numpy")

    def evaluate(self, x, y, x0, y0, Zc):
        """Evaluate the CBF at a batch of points.

        Args:
            x (np.array): x coordinates of the points (normalized), size N
            y (np.array): y coordinates of the points (normalized), size N
            x0 (float): x coordinate of the center of the sphere (normalized)
            y0 (float): y coordinate of the center of the sphere (normalized)
            Zc (float): z coordinate of the center of the sphere (camera frame)

        Returns:
            np.array: the values of the CBF at the given points, size N
        """
        if isinstance(x, float):
            return self.h(x, y, x0, y0, Zc).squeeze()

        h_vals = np.zeros(len(x))
        for i in range(len(x)):
            h_vals[i] = self.h(x[i], y[i], x0, y0, Zc).squeeze()
        return h_vals
    
    def evaluate_gradient(self, x, y, x0, y0, Zc):
        """Evaluate the gradient of the CBF at a batch of points.

        Args:
            x (np.array): x coordinates of the points (normalized), size N
            y (np.array): y coordinates of the points (normalized), size N
            x0 (float): x coordinate of the center of the sphere (normalized)
            y0 (float): y coordinate of the center of the sphere (normalized)
            Zc (float): z coordinate of the center of the sphere (camera frame)

        Returns:
            np.array: the gradient of the CBF at the given points, size Nx5
        """
        if isinstance(x, float):
            return self.partial_h(x, y, x0, y0, Zc).squeeze()

        partial_h_vals = np.zeros((len(x), 5))
        for i in range(len(x)):
            partial_h_vals[i,:] = self.partial_h(x[i], y[i], x0, y0, Zc).squeeze()
        return partial_h_vals
    
  