import kfp
from kfp import dsl
from kfp.components import func_to_container_op

# Import functions from other files
from load_data import *
from knn_train import *
from knn_inference import *
from rf_train import *
from rf_inference import *

@dsl.pipeline(name='somospiepipeline', description='Pipeline for somospie')
def pipeline(
    container_image: str ="olayap/somospie-gdal"
):

    # Create a PVC where input data is stored
    pvc_op = dsl.VolumeOp(name="odh-pvc",
                           resource_name="pvc-odh-oklahoma1km",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": "odh-oklahoma1km", 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO)


    # How to define a component: https://kubeflow-pipelines.readthedocs.io/en/stable/source/kfp.components.html#kfp.components.create_component_from_func
    data_op = kfp.components.create_component_from_func(load_data, base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    knn_train_op = kfp.components.create_component_from_func(knn_train,  base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    knn_inference_op = kfp.components.create_component_from_func(knn_inference,  base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    rf_train_op = kfp.components.create_component_from_func(rf_train,  base_image =str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    rf_inference_op = kfp.components.create_component_from_func(rf_inference,  base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    


     # Get data and split in train and validation
    data_task = data_op("/cos/train.csv", "/cos/", "/cos/data.json").add_pvolumes({"/cos/": pvc_op.volume})
    # Train after splitting the data
    ## KNN:
    knn_train_task = knn_train_op(data_task.outputs['data'], 20, 3, "/cos/model_knn.pkl").add_pvolumes({"/cos/": pvc_op.volume})
    rf_train_task = rf_train_op(data_task.outputs['data'], 2000, 3, "/cos/model_rf.pkl").add_pvolumes({"/cos/": pvc_op.volume})

    n_tiles = 1
    # Inference on multiple tiles
    for i in range(n_tiles):
        tile_id = "tile_{0:04d}".format(i) ## Check format to read
        ## KNN:
        knn_inference_task = knn_inference_op(knn_train_task.output, data_task.outputs['scaler'], "/cos/"+tile_id+".tif", 
                            "/out_knn/"+tile_id+".tif").add_pvolumes({"/cos/": pvc_op.volume, "/out_knn/": pvc_op.volume})
        ## RF:
        rf_inference_task = rf_inference_op(rf_train_task.output, data_task.outputs['scaler'], "/cos/"+tile_id+".tif", 
                            "/out_rf/"+tile_id+".tif").add_pvolumes({"/cos/": pvc_op.volume, "/out_rf/": pvc_op.volume})

    #Gather results for analysis stage


if __name__ == '__main__':
    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'pipeline.yaml')
 