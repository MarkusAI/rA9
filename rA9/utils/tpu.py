import os
import requests
from jax.config import config

'''
def TPU_setup_cloud(addr):
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
    print(config.FLAGS.jax_backend_target)
   ''' 
    
    
def TPU_setup_colab():
    # Make sure the Colab Runtime is set to Accelerator: TPU.
    if 'TPU_DRIVER_MODE' not in globals():
        url = 'http://' + os.environ['COLAB_TPU_ADDR'].split(':')[0] + ':8475/requestversion/tpu_driver_nightly'
        resp = requests.post(url)
        TPU_DRIVER_MODE = 1

    # The following is required to use TPU Driver as JAX's backend.
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
    print(config.FLAGS.jax_backend_target)
