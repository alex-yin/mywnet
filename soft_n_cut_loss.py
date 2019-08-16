import numpy as np
import tensorflow as tf

# done: make this batch wise
def edge_weights(flatten_image, rows, cols, std_intensity=3, std_position=1, radius=10):
        '''
        Inputs :
        flatten_image [B, m]: 2 dim tf array of the row flattened image ( intensity is the average of the three channels)
        std_intensity : standard deviation for intensity

        Output :
        weights [B, m, m]:  3d tf array edge weights in the pixel graph
        '''
        A = outer_product(flatten_image, tf.ones_like(flatten_image)) # [B, m, m]
        A_T = tf.transpose(A, [0,2,1]) # [B, m, m]
        intensity_weight = tf.exp(- ((A - A_T) / std_intensity)**2) # [B, m, m]

        xx, yy = tf.meshgrid(tf.range(rows), tf.range(cols))
        xx = tf.reshape(xx, (rows*cols,))
        yy = tf.reshape(yy, (rows*cols,))
        A_x = outer_product(xx, tf.ones_like(xx))
        A_y = outer_product(yy, tf.ones_like(yy))

        xi_xj = A_x - tf.transpose(A_x)
        yi_yj = A_y - tf.transpose(A_y)

        sq_distance_matrix = tf.square(xi_xj) + tf.square(yi_yj)

        dist_weight = tf.exp(-tf.divide(sq_distance_matrix,tf.square(std_position)))
        dist_weight = tf.cast(dist_weight, tf.float32)
        weights = intensity_weight * dist_weight

        return weights

# done: make this batch wise
def outer_product(v1,v2):
        '''
        Inputs:
        v1 : [B, m] tf array
        v2 : [B, m] tf array

        Output :
        v1 x v2 : [B, m, m] array
        '''
        v1 = tf.squeeze(v1) # [B, m]
        v2 = tf.squeeze(v2) # [B, m]
        v1 = tf.expand_dims((v1), axis=-1) # [B,m,1]
        v2 = tf.expand_dims((v2), axis=-2) # [B,1,m]
        return tf.matmul(v1, v2) #[B, m, m]


# done: make this batch wise
def numerator(k_class_prob, weights):

        '''
        Inputs :
        k_class_prob [B, m]: k_class flatten pixelwise probability tensor
        weights [B, m, m]: edge weights tensor
        '''
        return tf.reduce_sum(tf.multiply(weights, outer_product(k_class_prob,k_class_prob)), axis=[1,2]) # [B]

# done: make this batch wise
def denominator(k_class_prob, weights):
        '''
        Inputs:
        k_class_prob [B, m]: k_class flatten pixelwise probability tensor
        weights [B, m, m]: edge weights tensor
        '''
        # done: sum the weights first
        sum_w = tf.reduce_sum(weights, axis=-1) #[B, m]
        # [B, 1, m] @ [B, m, 1] -> [B, 1, 1]
        return tf.squeeze(tf.matmul(tf.expand_dims(k_class_prob, axis=1), tf.expand_dims(sum_w, axis=-1))) # [B]
        # return tf.reduce_sum(tf.multiply(weights, outer_product(k_class_prob, tf.ones(tf.shape(k_class_prob)))))

# done: make this batch wise
def soft_n_cut_loss(image, prob, k, rows, cols):
        '''
        Inputs:
        prob [B, h, w, k]: segmentation tensor
        k scalar: number of classes (integer)
        image [B, h, w]: 1-channel image

        Output :
        soft_n_cut_loss tensor for a single image

        '''
        batch_size = tf.shape(image)[0]
        # [B, h, w] -> [B, m]
        flatten_image = tf.reshape(image, [batch_size, -1])
        # done: fix this make it work with batch
        soft_n_cut_loss = tf.ones([batch_size]) * k # [B]
        weights = edge_weights(flatten_image, rows, cols) # [B, m, m]

        for t in range(k):
            # [B, h, w] -> [B, m]
            k_class_prob = tf.reshape(prob[...,t], [batch_size, -1])
            dis_assoc = numerator(k_class_prob, weights) # [B]
            assoc = denominator(k_class_prob, weights) # [B]
            soft_n_cut_loss = soft_n_cut_loss - (dis_assoc / assoc) #[B]

        return soft_n_cut_loss
