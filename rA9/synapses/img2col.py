import jax.numpy as jnp


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = jnp.repeat(jnp.arange(field_height), field_width)
    i0 = jnp.tile(i0, C)
    i1 = stride * jnp.repeat(jnp.arange(out_height), out_width)
    j0 = jnp.tile(jnp.arange(field_width), field_height * C)
    j1 = stride * jnp.tile(jnp.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = jnp.array(x[:, k, i, j])
    C = x.shape[1]
    cols = jnp.transpose(cols, (1, 2, 0))
    cols = cols.reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = jnp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    jnp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return jnp.array(x_padded[:, :, padding:-padding, padding:-padding])
