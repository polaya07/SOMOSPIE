import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from typing import NamedTuple


 Import functions from other files
from merge import *
from reproject import *
from crop import *
from compute import *

@dsl.pipeline(name='somospie data generation pipeline', description='Pipeline for somospie data generation')
def pipeline(container_image:str):

    # Create a PVC where input data is stored
    pvc_op = dsl.VolumeOp(name="pvc-geotiled",
                           resource_name="pvc-goetiled",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": "pvc-goetiled", 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO)


    # How to define a component: https://kubeflow-pipelines.readthedocs.io/en/stable/source/kfp.components.html#kfp.components.create_component_from_func
    merge_op = kfp.components.create_component_from_func(merge, base_image = container_image)#packages_to_install=["numpy", "pandas", "scikit-learn"])
    reproject_op = kfp.components.create_component_from_func(reproject,  base_image = container_image)#packages_to_install=["numpy", "pandas", "scikit-learn"])
    crop_op = kfp.components.create_component_from_func(crop,  base_image = container_image)#packages_to_install=["numpy", "pandas", "scikit-learn"])
    compute_op = kfp.components.create_component_from_func(compute,  base_image =container_image)#packages_to_install=["numpy", "pandas", "scikit-learn"])     


     # Get data and split in train and validation
    data_task = data_op("/cos/").add_pvolumes({"/cos/": pvc_op.volume})
    # Train and Test after splitting the data
    ## KNN:
    knn_train_task = knn_train_op(data_task.output, 20, 3).add_pvolumes({"/cos/": pvc_op.volume})
    knn_inference_task = knn_inference_op(knn_train_task.output).add_pvolumes({"/cos/": pvc_op.volume})
    ## RF:
    rf_train_task = rf_train_op(data_task.output, 2000, 3).add_pvolumes({"/cos/": pvc_op.volume})
    rf_inference_task = rf_inference_op(rf_train_task.output).add_pvolumes({"/cos/": pvc_op.volume})

if __name__ == '__main__':
    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'pipeline.yaml')