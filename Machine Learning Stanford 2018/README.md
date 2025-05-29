# CS229 Autumn 2018

All lecture notes, slides and assignments for [CS229: Machine Learning](http://cs229.stanford.edu/) course by Stanford University.

The videos of all lectures are available [on YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU).

Useful links:
- [CS229 Summer 2019 edition](https://github.com/maxim5/cs229-2019-summer)


Vd: X= [1 2 3
        4 5 6
        7 8 9
        3 7 3]
    X~ MxN
Note: Always imagine Vector is a column vector (Nx1)
    X[i] is also a column vector (Nx1)
    y= [0
        1
        0
        0]
    y~ Mx1
    y.transpose()~ 1xM
Note: Matrix and vector manipulation: Nối đuôi nhau
VD: (MxN).dot((NxD)) ~ MxD
    (NxM) @ (Mx1) ~ (Nx1)

VD : y_pred[i] = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y).T.dot(x[i])
                                (MxN).T  (MxM)       (MxN)         (NxM)   (MxM)       (Mx1).T     (Nx1)
                                 (NxM)   (MxM)       (MxN)         (NxM)   (MxN)       (Mx1).T     (Nx1)
                                 Compute from left to right even .T                          ^
                                                                phải thực hiện đủ các ma trận ở trước rồi mới transpose()
                        