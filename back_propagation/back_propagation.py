import plot_result
import numpy as np

def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def loss(y, y_hat):
    return np.mean((y-y_hat) ** 2)

def derivative_loss(y, y_hat):
    return (y-y_hat)*(2/y.shape[0])

class layer():
    def __init__(self, input_size, output_size):
        self.W = np.random.normal(0, 1, (input_size+1, output_size))
        
    def forward(self, x):
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        self.forward_gradient = x
        self.y = sigmoid(np.matmul(x, self.W))
        return self.y
    
    def backward(self, d_l):
        self.backward_gradient = np.multiply(derivative_sigmoid(self.y), d_l)
        return np.matmul(self.backward_gradient, self.W[:-1].T)
    
    def update(self, learning_rate):
        self.gradient = np.matmul(self.forward_gradient.T, self.backward_gradient)
        self.W -= learning_rate * self.gradient
        return self.gradient
        

class NN():
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.1):
        self.learning_rate = learning_rate
        sizes = [input_size, hidden_size, hidden_size, output_size]
        sizes2 = [hidden_size, hidden_size, output_size, 0]
        self.layers = []
        
        for s1, s2 in zip(sizes, sizes2):
            if (s1+1)*s2 == 0:
                continue
            self.layers += [layer(s1, s2)]
        
    def forward_pass(self, x):
        _x = x
        for layer in self.layers:
            _x = layer.forward(_x)
        return _x
    
    def backward_pass(self, d_l):
        _d_l = d_l
        for layer in self.layers[::-1]:
            _d_l = layer.backward(_d_l)
            
    def update(self):
        gradients = []
        for layer in self.layers:
            gradients += [layer.update(self.learning_rate)]

if __name__ == '__main__':
    NN_linear = NN(2, 4, 1, 1)
    NN_XOR = NN(2, 4, 1, 1)
    
    loss_threshold = 0.005
    linear_stop = False
    xor_stop = False
    x_linear, y_linear = generate_linear()
    x_xor, y_xor = generate_XOR_easy()
    
    #tranning
    for i in range(10000):
        if not linear_stop:
            y = NN_linear.forward_pass(x_linear)
            loss_linear = loss(y, y_linear)
            NN_linear.backward_pass(derivative_loss(y, y_linear))
            NN_linear.update()
            
            if(loss_linear < loss_threshold):
                linear_stop = True
                
        if not xor_stop:
            y = NN_XOR.forward_pass(x_xor)
            loss_xor = loss(y, y_xor)
            NN_XOR.backward_pass(derivative_loss(y, y_xor))
            NN_XOR.update()

        if i%200 == 0 or (linear_stop and xor_stop):
            print ('[{:4d}] linear loss : {:4f} \t XOR loss : {:.4f}'.format(i, loss_linear, loss_xor))
        
        if linear_stop and xor_stop:
            break
        
    y1 = NN_linear.forward_pass(x_linear)
    y2 = NN_XOR.forward_pass(x_xor)
    plot_result.show_result(x_linear, y_linear, y1)
    plot_result.show_result(x_xor, y_xor, y2)
        
        