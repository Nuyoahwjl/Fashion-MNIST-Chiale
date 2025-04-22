import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # He 初始化
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros((out_channels, 1))
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)
        # 动量项
        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        F, _, HH, WW = self.weights.shape
        H_out = 1 + (H + 2 * self.padding - HH) // self.stride
        W_out = 1 + (W + 2 * self.padding - WW) // self.stride
        col = im2col(X, HH, WW, self.stride, self.padding)
        col_W = self.weights.reshape(F, -1).T
        out = np.dot(col, col_W) + self.bias.T
        out = out.reshape(N, H_out, W_out, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        N, F, H_out, W_out = dout.shape
        _, C, HH, WW = self.weights.shape
        H = (H_out - 1) * self.stride + HH - 2 * self.padding
        W = (W_out - 1) * self.stride + WW - 2 * self.padding
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, F)
        col = im2col(self.X, HH, WW, self.stride, self.padding)
        self.dweights = np.dot(col.T, dout).transpose(1, 0).reshape(F, C, HH, WW)
        self.dbias = np.sum(dout, axis=0, keepdims=True).T
        dcol = np.dot(dout, self.weights.reshape(F, -1))
        dX = col2im(dcol, self.X.shape, HH, WW, self.stride, self.padding)
        return dX

    def update_params(self, learning_rate, momentum=0.9):
        self.v_weights = momentum * self.v_weights - learning_rate * self.dweights
        self.weights += self.v_weights
        self.v_bias = momentum * self.v_bias - learning_rate * self.dbias
        self.bias += self.v_bias


class BatchNormalization:
    def __init__(self, channels):
        self.gamma = np.ones((1, channels, 1, 1))
        self.beta = np.zeros((1, channels, 1, 1))
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.moving_mean = np.zeros((1, channels, 1, 1))
        self.moving_var = np.ones((1, channels, 1, 1))
        self.eps = 1e-5
        self.momentum = 0.9

    def forward(self, x, train_flg=True):
        self.train_flg = train_flg
        N, C, H, W = x.shape
        if train_flg:
            mu = x.mean(axis=(0, 2, 3), keepdims=True)
            xc = x - mu
            var = np.mean(xc ** 2, axis=(0, 2, 3), keepdims=True)
            std = np.sqrt(var + self.eps)
            xn = xc / std
            self.xc = xc
            self.xn = xn
            self.std = std
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mu
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var
        else:
            xc = x - self.moving_mean
            xn = xc / np.sqrt(self.moving_var + self.eps)
        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        N, C, H, W = dout.shape
        dbeta = dout.sum(axis=(0, 2, 3), keepdims=True)
        dgamma = np.sum(self.xn * dout, axis=(0, 2, 3), keepdims=True)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=(0, 2, 3), keepdims=True)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / (N * H * W)) * self.xc * dvar
        dmu = np.sum(dxc, axis=(0, 2, 3), keepdims=True)
        dx = dxc - dmu / (N * H * W)
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx

    def update_params(self, learning_rate):
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        H_out = 1 + (H - self.pool_size) // self.stride
        W_out = 1 + (W - self.pool_size) // self.stride
        out = np.zeros((N, C, H_out, W_out))
        self.arg_max = np.zeros((N, C, H_out, W_out), dtype=np.int64)

        for i in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        out[i, c, h, w] = np.max(X[i, c, h_start:h_end, w_start:w_end])
                        self.arg_max[i, c, h, w] = np.argmax(X[i, c, h_start:h_end, w_start:w_end])
        return out

    def backward(self, dout):
        N, C, H_out, W_out = dout.shape
        _, _, H, W = self.X.shape
        dX = np.zeros_like(self.X)
        for i in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        idx = self.arg_max[i, c, h, w]
                        dX[i, c, h_start + idx // self.pool_size, w_start + idx % self.pool_size] = dout[i, c, h, w]
        return dX


class FullyConnected:
    def __init__(self, input_size, output_size):
        # He 初始化
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)
        # 动量项
        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)

    def forward(self, X):
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, dout):
        dX = np.dot(dout, self.weights.T)
        self.dweights = np.dot(self.X.T, dout)
        self.dbias = np.sum(dout, axis=0, keepdims=True)
        return dX

    def update_params(self, learning_rate, momentum=0.9):
        self.v_weights = momentum * self.v_weights - learning_rate * self.dweights
        self.weights += self.v_weights
        self.v_bias = momentum * self.v_bias - learning_rate * self.dbias
        self.bias += self.v_bias


class Softmax:
    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def backward(self, y_pred, y_true):
        N = y_pred.shape[0]
        return (y_pred - y_true) / N


class CNN:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.bn1 = BatchNormalization(6)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.bn2 = BatchNormalization(16)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        self.fc1 = FullyConnected(input_size=5 * 5 * 16, output_size=120)
        self.relu3 = ReLU()
        self.fc2 = FullyConnected(input_size=120, output_size=84)
        self.relu4 = ReLU()
        self.fc3 = FullyConnected(input_size=84, output_size=10)
        self.softmax = Softmax()

    def forward(self, X):
        X = X.reshape(-1, 1, 28, 28)
        out = self.conv1.forward(X)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        out = self.fc2.forward(out)
        out = self.relu4.forward(out)
        out = self.fc3.forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self, y_pred, y_true):
        dout = self.softmax.backward(y_pred, y_true)
        dout = self.fc3.backward(dout)
        dout = self.relu4.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout)
        dout = dout.reshape(-1, 16, 5, 5)
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        return dout

    def update_params(self, learning_rate, momentum=0.9):
        self.conv1.update_params(learning_rate, momentum)
        self.bn1.update_params(learning_rate)
        self.conv2.update_params(learning_rate, momentum)
        self.bn2.update_params(learning_rate)
        self.fc1.update_params(learning_rate, momentum)
        self.fc2.update_params(learning_rate, momentum)
        self.fc3.update_params(learning_rate, momentum)

    def save_model(self, filename):
        model_params = {
            'conv1_weights': self.conv1.weights,
            'conv1_bias': self.conv1.bias,
            'bn1_gamma': self.bn1.gamma,
            'bn1_beta': self.bn1.beta,
            'conv2_weights': self.conv2.weights,
            'conv2_bias': self.conv2.bias,
            'bn2_gamma': self.bn2.gamma,
            'bn2_beta': self.bn2.beta,
            'fc1_weights': self.fc1.weights,
            'fc1_bias': self.fc1.bias,
            'fc2_weights': self.fc2.weights,
            'fc2_bias': self.fc2.bias,
            'fc3_weights': self.fc3.weights,
            'fc3_bias': self.fc3.bias,
        }
        np.savez(filename, **model_params)

    def load_model(self, filename):
        model_params = np.load(filename)
        self.conv1.weights = model_params['conv1_weights']
        self.conv1.bias = model_params['conv1_bias']
        self.bn1.gamma = model_params['bn1_gamma']
        self.bn1.beta = model_params['bn1_beta']
        self.conv2.weights = model_params['conv2_weights']
        self.conv2.bias = model_params['conv2_bias']
        self.bn2.gamma = model_params['bn2_gamma']
        self.bn2.beta = model_params['bn2_beta']
        self.fc1.weights = model_params['fc1_weights']
        self.fc1.bias = model_params['fc1_bias']
        self.fc2.weights = model_params['fc2_weights']
        self.fc2.bias = model_params['fc2_bias']
        self.fc3.weights = model_params['fc3_weights']
        self.fc3.bias = model_params['fc3_bias']
    
    def print_model(self):
        layers = [
            (self.conv1, 'Conv2D', 'conv1'),
            (self.bn1, 'BatchNorm', 'bn1'),
            (self.relu1, 'ReLU', 'relu1'),
            (self.pool1, 'MaxPool2D', 'pool1'),
            (self.conv2, 'Conv2D', 'conv2'),
            (self.bn2, 'BatchNorm', 'bn2'),
            (self.relu2, 'ReLU', 'relu2'),
            (self.pool2, 'MaxPool2D', 'pool2'),
            (self.fc1, 'FullyConnected', 'fc1'),
            (self.relu3, 'ReLU', 'relu3'),
            (self.fc2, 'FullyConnected', 'fc2'),
            (self.relu4, 'ReLU', 'relu4'),
            (self.fc3, 'FullyConnected', 'fc3'),
        ]
        current_shape = (1, 28, 28)  # 输入形状: (channels, height, width)
        total_params = 0
        print("Layer (type)         Output Shape         Param #")
        print("===================================================")
        for layer_info in layers:
            layer_obj, layer_type, layer_name = layer_info
            params = 0
            output_shape = current_shape
            if layer_type == 'Conv2D':
                in_channels, H_in, W_in = current_shape
                padding = layer_obj.padding
                kernel_size = layer_obj.kernel_size
                stride = layer_obj.stride
                out_channels = layer_obj.out_channels                
                H_out = (H_in + 2 * padding - kernel_size) // stride + 1
                W_out = (W_in + 2 * padding - kernel_size) // stride + 1
                output_shape = (out_channels, H_out, W_out)
                params = (in_channels * kernel_size**2) * out_channels + out_channels
                layer_desc = f"Conv2D"
            elif layer_type == 'BatchNorm':
                channels = current_shape[0]
                params = 2 * channels  # gamma和beta
                layer_desc = f"BatchNorm"
                output_shape = current_shape
            elif layer_type == 'ReLU':
                layer_desc = "ReLU"
                params = 0
                output_shape = current_shape
            elif layer_type == 'MaxPool2D':
                pool_size = layer_obj.pool_size
                stride = layer_obj.stride
                channels, H_in, W_in = current_shape
                H_out = (H_in - pool_size) // stride + 1
                W_out = (W_in - pool_size) // stride + 1
                output_shape = (channels, H_out, W_out)
                layer_desc = f"MaxPool2D"
                params = 0
            elif layer_type == 'FullyConnected':
                if len(current_shape) == 3:
                    input_dim = current_shape[0] * current_shape[1] * current_shape[2]
                else:
                    input_dim = current_shape[0]
                output_dim = layer_obj.weights.shape[1]
                params = input_dim * output_dim + output_dim
                output_shape = (output_dim,)
                layer_desc = f"FullyConnected"
            # 格式化输出
            print(f"{layer_desc.ljust(20)} {str(output_shape).ljust(20)} {params}")
            total_params += params
            current_shape = output_shape
        print("===================================================")
        print(f"Total params: {total_params}")


if __name__ == "__main__":
    model = CNN()
    model.print_model()