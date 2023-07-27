import kfp
from kfp import dsl
from kfp.components import func_to_container_op
import pandas as pd

# Import functions from other files
from load_data import *
from knn_train import *
from rf_train import *
from inference import *
#from tif2df import *

@dsl.pipeline(name='somospiepipeline', description='Pipeline for somospie')
def pipeline(
    container_image: str ="icr.io/somospie/somospie-gdal-netcdf",
    total_tiles: int = 3,
    eval_cos: str = 'tn-30m-eval', #"oklahoma-30m-eval",
    train_cos: str = 'tn-27km', #"oklahoma-27km",
    predict_cos: str = 'tn-30m-pred', #"p-oklahoma-30m",
    mem_lim: int = 13,
    random_seed: int = 3,
    k_neighbors: int = 20,
    trees: int = 2000

):

    # Create a PVCs
    # Evaluation data
    pvc_eval = dsl.VolumeOp(name="pvc-eval",
                           resource_name="pvc-odh-eval",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": eval_cos, 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO).set_retry(3)

    # Train data
    pvc_train = dsl.VolumeOp(name="pvc-train",
                           resource_name="pvc-odh-train",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": train_cos, 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO).set_retry(3)
    
    # Predictions data
    pvc_predictions = dsl.VolumeOp(name="pvc-predictions",
                           resource_name="pvc-odh-predictions",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": predict_cos, 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO).set_retry(3)


    # How to define a component: https://kubeflow-pipelines.readthedocs.io/en/stable/source/kfp.components.html#kfp.components.create_component_from_func
    data_op = kfp.components.create_component_from_func(load_data, base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    knn_train_op = kfp.components.create_component_from_func(knn_train,  base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    rf_train_op = kfp.components.create_component_from_func(rf_train,  base_image =str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    inference_op = kfp.components.create_component_from_func(inference,  base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    #tif2df_op = kfp.components.create_component_from_func(tif2df,  base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])

    region_res: str = "tn-30m"
    year = 2010
    months = 3
    
    for month in range(1,months+1):
        train_file = "/train/"+'{0:04d}_{1:02d}.tif'.format(year, month)
        temp_data_json = "/train/"+'{0:04d}_{1:02d}.json'.format(year, month)
        # Get data and split in train and validation
        data_task = data_op(train_file, "/train/", temp_data_json).add_pvolumes({"/train/": pvc_train.volume}).set_retry(3)
        # Train after splitting the data
        ## KNN:
        knn_train_task = knn_train_op(data_task.outputs['data'], k_neighbors, random_seed, "/train/model_knn"+str(month)+".pkl").add_pvolumes({"/train/": pvc_train.volume}).set_retry(3)
        rf_train_task = rf_train_op(data_task.outputs['data'], trees, random_seed, "/train/model_rf"+str(month)+".pkl").add_pvolumes({"/train/": pvc_train.volume}).set_retry(3)

        total_tiles = 4 # 9 for Oklahoma. Here is the total tiles
        band_names = ['elevation', 'aspect', 'slope', 'twi']
        #tif2df_task =[]
        # Inference on multiple tiles
        for i in range(total_tiles):
            #tile_id = "midwest_region_evaluation_10m_grid_{0:03d}".format(i) ## Check format to read
            tile_id = region_res+"_eval_{0:04d}".format(i)
            ## Read Data:
            #tif2df_task.append(tif2df_op("/eval/"+tile_id+".csv", band_names))
            ## KNN:
            knn_inference_task = inference_op(knn_train_task.output, data_task.outputs['scaler'], "/eval/"+tile_id+".tif" , "/out_knn/knn/",
                                "/out_knn/knn/"+tile_id+".tif","/out_knn/knn/"+tile_id+".csv").add_pvolumes({"/train/": pvc_train.volume, "/eval/": pvc_eval.volume, 
                                "/out_knn/": pvc_predictions.volume}).set_retry(3).set_memory_request('20G').add_node_selector_constraint("ibm-cloud.kubernetes.io/worker-pool-name", "inference")
            ## RF:
            rf_inference_task = inference_op(rf_train_task.output, data_task.outputs['scaler'],  "/eval/"+tile_id+".tif" , "/out_rf/rf/",
                                "/out_rf/rf/"+tile_id+".tif", "/out_rf/rf/"+tile_id+".csv").add_pvolumes({"/train/": pvc_train.volume, "/eval/": pvc_eval.volume, 
                                "/out_rf/": pvc_predictions.volume}).set_retry(3).set_memory_request('20G').add_node_selector_constraint("ibm-cloud.kubernetes.io/worker-pool-name", "inference")

        #Gather results for analysis stage


if __name__ == '__main__':
    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'ml-model-pipeline/pipeline.yaml')
 