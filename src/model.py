import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Multiply, Concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model

def conv_block(input_tensor, num_filters, use_residual=True):
    """Conv Block with optional residual connection"""
    shortcut = input_tensor

    # First conv layer
    x = Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second conv layer
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)

    # Optional residual connection
    if use_residual:
        if input_tensor.shape[-1] != num_filters:
            shortcut = Conv2D(num_filters, 1, padding="same")(shortcut)
        x = Add()([x, shortcut])
        
    x = Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters, use_residual=True):
    """Encoder block with optional residual connections and pooling"""
    x = conv_block(input_tensor, num_filters, use_residual=use_residual)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def attention_gate(skip_feature, gating_signal, num_filters):
    """
    Attention gate to focus on relevant features from skip connections
    """
    # Transform skip connection
    skip = Conv2D(num_filters, 1, padding="same")(skip_feature)
    skip = BatchNormalization()(skip)

    # Transform gating signal
    gate = Conv2D(num_filters, 1, padding="same")(gating_signal)
    gate = BatchNormalization()(gate)

    # Combine and apply activation
    combined = Add()([skip, gate])
    combined = Activation('relu')(combined)

    # Generate attention coefficients
    attention = Conv2D(1, 1, padding="same")(combined)
    attention = Activation('sigmoid')(attention)

    # Apply attention to skip features
    attended_features = Multiply()([skip_feature, attention])
    return attended_features

def decoder_block(input_tensor, skip_feature, num_filters, use_attention=True, use_residual=True):
    """Decoder block with optional attention gate and residual conv block"""
    # Upsample
    x = Conv2DTranspose(num_filters, 2, strides=2, padding='same')(input_tensor)

    # Optional attention gate
    if use_attention:
        skip_feature = attention_gate(skip_feature, x, num_filters)

    # Concatenate with skip features
    x = Concatenate()([x, skip_feature])

    # Apply conv block
    x = conv_block(x, num_filters, use_residual=use_residual)
    return x

def build_residual_attention_unet(n_classes, img_height, img_width, img_channels, dropout_rate=0.3, use_attention=True, use_residual=True):
    """Builds the U-Net model with optional Attention and Residual components"""
    inputs = Input((img_height, img_width, img_channels))

    # Encoder
    s1, p1 = encoder_block(inputs, 64, use_residual=use_residual)
    s1 = Dropout(dropout_rate)(s1)
    s2, p2 = encoder_block(p1, 128, use_residual=use_residual)
    s2 = Dropout(dropout_rate)(s2)
    s3, p3 = encoder_block(p2, 256, use_residual=use_residual)
    s3 = Dropout(dropout_rate)(s3)
    s4, p4 = encoder_block(p3, 512, use_residual=use_residual)
    s4 = Dropout(dropout_rate)(s4)

    # Bridge
    b1 = conv_block(p4, 1024, use_residual=use_residual)
    b1 = Dropout(dropout_rate)(b1)

    # Decoder
    d1 = decoder_block(b1, s4, 512, use_attention=use_attention, use_residual=use_residual)
    d1 = Dropout(dropout_rate)(d1)
    d2 = decoder_block(d1, s3, 256, use_attention=use_attention, use_residual=use_residual)
    d2 = Dropout(dropout_rate)(d2)
    d3 = decoder_block(d2, s2, 128, use_attention=use_attention, use_residual=use_residual)
    d3 = Dropout(dropout_rate)(d3)
    d4 = decoder_block(d3, s1, 64, use_attention=use_attention, use_residual=use_residual)
    d4 = Dropout(dropout_rate)(d4)

    # Output layer
    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax")(d4)
    
    # Clean naming
    name_parts = []
    if use_residual: name_parts.append("Residual")
    if use_attention: name_parts.append("Attention")
    name_parts.append("U-Net")
    
    model = Model(inputs, outputs, name="-".join(name_parts))
    return model
