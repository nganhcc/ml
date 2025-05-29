import numpy as np

def backward_max_pool(data, pool_width, pool_height, output_grad):
    """
    Compute the gradient of the loss with respect to the data in the max pooling layer.

    data is of the shape (# channels, width, height)
    output_grad is of shape (# channels, width // pool_width, height // pool_height)

    output_grad is the gradient of the loss with respect to the output of the backward max
    pool layer.

    Returns:
        The gradient of the loss with respect to the data (of same shape as data)
    """
    """
    Compute the gradient of the loss with respect to the data in the max pooling layer.

    data is of the shape (# channels, width, height)
    output_grad is of shape (# channels, width // pool_width, height // pool_height)

    output_grad is the gradient of the loss with respect to the output of the backward max
    pool layer.

    Returns:
        The gradient of the loss with respect to the data (of same shape as data)
    """
    
    # *** START CODE HERE ***
    input_channels, input_width, input_height= data.shape
    pooling_data_grad=np.zeros(data.shape)
    for x in range (0,input_width,pool_width):
        for y in range(0,pool_height,input_height):
            for k in range(input_channels):
                tmp=data[k,x: (x+pool_width), y: (y+pool_height)]
                flat_index=np.argmax(tmp)
                row, col=np.unravel_index(flat_index, tmp.shape)
                pooling_data_grad[k, x+row, y+col]=output_grad[k, x//pool_width,y//pool_height]
    return pooling_data_grad
    input_channels, input_width, input_height= data.shape
    pooling_data_grad=np.zeros(data.shape)
    for x in range (0,input_width,pool_width):
        for y in range(0,input_height,pool_height):
            for k in range(input_channels):
                tmp=data[k,x: (x+pool_width), y: (y+pool_height)]
                pooling_data_grad[k, x: (x+pool_width), y:(y+pool_height)][np.unravel_index(tmp.argmax(),tmp.shape)]=output_grad[k, x//pool_width,y//pool_height]
    return pooling_data_grad
    # *** START CODE HERE ***
def main():
    data=np.arange(32).reshape(2,4,4)
    grad=np.random.random((2,2,2))
    print(backward_max_pool(data,2,2, grad))
if __name__=="__main__":
    main()