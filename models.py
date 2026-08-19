"""
DeHazeX Pro - Model Definitions (TensorFlow / Keras)
-----------------------------------------------------
Three architectures, matching the project report:
    1. AOD-Net        -> lightweight CNN, fastest
    2. Attention U-Net -> proposed hybrid model, best PSNR/SSIM
    3. DehazeGAN       -> Generator (ResNet-style) + PatchGAN Discriminator

All models take a 256x256x3 image normalised to [0,1] and output a
256x256x3 image in [0,1] (dehazed).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# 1. AOD-NET  (Li et al., ICCV 2017)
#    J(x) = K(x) * I(x) - K(x) + 1
# ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="DeHazeX")
class AODFormula(layers.Layer):
    """J(x) = K(x) * I(x) - K(x) + 1, clipped to [0,1].

    Implemented as a proper Layer (not a Lambda-with-python-function) so the
    saved .keras file stores plain config, not pickled Python bytecode. This
    avoids both the Keras 'unsafe Lambda deserialization' block and bytecode
    incompatibility across different Python versions.
    """
    def call(self, inputs):
        k_map, hazy = inputs
        out = k_map * hazy - k_map + 1.0
        return tf.clip_by_value(out, 0.0, 1.0)


def build_aod_net(input_shape=(256, 256, 3)):
    inp = layers.Input(shape=input_shape, name="hazy_input")

    conv1 = layers.Conv2D(3, 1, padding="same", activation="relu", name="conv1")(inp)
    conv2 = layers.Conv2D(3, 3, padding="same", activation="relu", name="conv2")(conv1)

    concat1 = layers.Concatenate(name="concat1")([conv1, conv2])
    conv3 = layers.Conv2D(3, 5, padding="same", activation="relu", name="conv3")(concat1)

    concat2 = layers.Concatenate(name="concat2")([conv2, conv3])
    conv4 = layers.Conv2D(3, 7, padding="same", activation="relu", name="conv4")(concat2)

    concat3 = layers.Concatenate(name="concat3")([conv1, conv2, conv3, conv4])
    k = layers.Conv2D(3, 3, padding="same", activation="relu", name="k_estimate")(concat3)

    # J(x) = K(x)*I(x) - K(x) + 1   (b is fixed to 1, as in the paper)
    out = AODFormula(name="aod_output")([k, inp])
    return Model(inp, out, name="AOD_Net")


# ---------------------------------------------------------------------------
# 2. ATTENTION U-NET  (proposed hybrid model)
# ---------------------------------------------------------------------------
def _conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def _attention_gate(g, x, filters, name):
    """g = decoder (gating) signal, x = encoder skip features."""
    theta_g = layers.Conv2D(filters, 1, padding="same", name=f"{name}_wg")(g)
    theta_x = layers.Conv2D(filters, 1, padding="same", name=f"{name}_wx")(x)
    add = layers.Add(name=f"{name}_add")([theta_g, theta_x])
    act = layers.Activation("relu", name=f"{name}_relu")(add)
    psi = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name=f"{name}_psi")(act)
    return layers.Multiply(name=f"{name}_gated")([x, psi])


def build_attention_unet(input_shape=(256, 256, 3)):
    inp = layers.Input(shape=input_shape, name="hazy_input")

    # Encoder
    e1 = _conv_block(inp, 64, "enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(e1)

    e2 = _conv_block(p1, 128, "enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(e2)

    e3 = _conv_block(p2, 256, "enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(e3)

    # Bottleneck
    b = _conv_block(p3, 512, "bottleneck")

    # Decoder 3 + Attention Gate 3
    up3 = layers.Conv2DTranspose(256, 2, strides=2, padding="same", name="up3")(b)
    a3 = _attention_gate(up3, e3, 128, "att3")
    d3 = layers.Concatenate(name="concat3")([up3, a3])
    d3 = _conv_block(d3, 256, "dec3")

    # Decoder 2 + Attention Gate 2
    up2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same", name="up2")(d3)
    a2 = _attention_gate(up2, e2, 64, "att2")
    d2 = layers.Concatenate(name="concat2")([up2, a2])
    d2 = _conv_block(d2, 128, "dec2")

    # Decoder 1 + Attention Gate 1
    up1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same", name="up1")(d2)
    a1 = _attention_gate(up1, e1, 32, "att1")
    d1 = layers.Concatenate(name="concat1")([up1, a1])
    d1 = _conv_block(d1, 64, "dec1")

    out = layers.Conv2D(3, 1, padding="same", activation="sigmoid", name="output")(d1)
    return Model(inp, out, name="Attention_UNet")


# ---------------------------------------------------------------------------
# 3. DEHAZE-GAN  (Generator = ResNet-style, Discriminator = PatchGAN)
# ---------------------------------------------------------------------------
def _residual_block(x, filters, name):
    skip = x
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.LayerNormalization(name=f"{name}_in1")(x)  # acts like Instance Norm per-sample
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.LayerNormalization(name=f"{name}_in2")(x)
    return layers.Add(name=f"{name}_skip")([skip, x])


def build_generator(input_shape=(256, 256, 3), n_residual=9):
    inp = layers.Input(shape=input_shape, name="hazy_input")

    x = layers.Conv2D(64, 7, padding="same", name="init_conv")(inp)
    x = layers.LayerNormalization(name="init_in")(x)
    x = layers.Activation("relu", name="init_relu")(x)

    # Downsample x2
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu", name="down1")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu", name="down2")(x)

    # 9 residual blocks
    for i in range(n_residual):
        x = _residual_block(x, 256, f"res{i+1}")

    # Upsample x2
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu", name="up1")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", name="up2")(x)

    out = layers.Conv2D(3, 7, padding="same", activation="sigmoid", name="gen_output")(x)
    return Model(inp, out, name="DehazeGAN_Generator")


def build_discriminator(input_shape=(256, 256, 3)):
    inp = layers.Input(shape=input_shape, name="image_input")
    x = inp
    for i, filters in enumerate([64, 128, 256, 512]):
        x = layers.Conv2D(filters, 4, strides=2, padding="same", name=f"d_conv{i+1}")(x)
        if i != 0:
            x = layers.LayerNormalization(name=f"d_in{i+1}")(x)
        x = layers.LeakyReLU(0.2, name=f"d_lrelu{i+1}")(x)
    patch_out = layers.Conv2D(1, 4, padding="same", name="patch_output")(x)  # PatchGAN map
    return Model(inp, patch_out, name="DehazeGAN_Discriminator")


MODEL_BUILDERS = {
    "aod_net": build_aod_net,
    "attention_unet": build_attention_unet,
    "gan_generator": build_generator,
}


if __name__ == "__main__":
    for name, builder in MODEL_BUILDERS.items():
        m = builder()
        print(f"\n{name}: {m.count_params():,} params")