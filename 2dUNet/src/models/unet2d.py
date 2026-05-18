"""
2D-UNet architecture for ocean subsurface reconstruction.

Encoder-decoder with skip connections. Supports configurable filter sizes,
dropout regularization, and L2 weight decay.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def double_conv_block(x, filters, dropout_rate=0.0, weight_decay=0.0):
    """Two consecutive Conv2D + BatchNorm + ReLU blocks with optional dropout."""
    reg = regularizers.l2(weight_decay) if weight_decay > 0 else None

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal",
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal",
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def downsample_block(x, filters, dropout_rate=0.0, weight_decay=0.0):
    """Conv block followed by 2x2 max pooling."""
    f = double_conv_block(x, filters, dropout_rate, weight_decay)
    p = layers.MaxPool2D(2)(f)
    return f, p


def upsample_block(x, conv_features, filters, dropout_rate=0.0, weight_decay=0.0):
    """Transpose conv for upsampling, concatenate with skip connection, then conv block."""
    reg = regularizers.l2(weight_decay) if weight_decay > 0 else None
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                               kernel_regularizer=reg)(x)
    x = layers.concatenate([x, conv_features])
    x = double_conv_block(x, filters, dropout_rate, weight_decay)
    return x


def build_unet2d(input_shape=(64, 64, 5), out_channels=52,
                 filters=[64, 128, 256, 512], dropout_rate=0.0, weight_decay=0.0):
    """
    Build a 2D-UNet model.

    Args:
        input_shape: (H, W, C) input tensor shape.
        out_channels: Number of output channels (e.g. 52 = 26 temp + 26 salt).
        filters: List of filter counts for each encoder stage.
        dropout_rate: Dropout probability after each conv block (0 = disabled).
        weight_decay: L2 regularization strength (0 = disabled).
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    f1, p1 = downsample_block(inputs, filters[0], dropout_rate, weight_decay)
    f2, p2 = downsample_block(p1, filters[1], dropout_rate, weight_decay)
    f3, p3 = downsample_block(p2, filters[2], dropout_rate, weight_decay)

    # Bottleneck
    bottleneck = double_conv_block(p3, filters[3], dropout_rate, weight_decay)

    # Decoder
    u1 = upsample_block(bottleneck, f3, filters[2], dropout_rate, weight_decay)
    u2 = upsample_block(u1, f2, filters[1], dropout_rate, weight_decay)
    u3 = upsample_block(u2, f1, filters[0], dropout_rate, weight_decay)

    # Output (no dropout/regularization on the final 1x1 conv)
    outputs = layers.Conv2D(out_channels, 1, padding="same", activation="linear")(u3)

    model = Model(inputs, outputs, name="unet2d")
    return model
