import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from kubernetes import client, config
from kubernetes.client import V1Volume, V1SecretVolumeSource, V1VolumeMount, V1HostPathVolumeSource


#@func_to_container_op
#def show_results(decision_tree : float, logistic_regression : float) -> None:
#    # Given the outputs from decision_tree and logistic regression components
#    # the results are shown.
#
#    print(f"Decision tree (accuracy): {decision_tree}")
#    print(f"Logistic regression (accuracy): {logistic_regression}")


@dsl.pipeline(name='somospiepipeline', description='Pipeline for somospie')
def pipeline():

    # Create a PVC where input data is stored
    data_op = dsl.VolumeOp(name="odh-pvc",
                           resource_name="pvc-odh-oklahoma1km",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": "odh-oklahoma1km", 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO)

    # Loads the yaml manifest for each component
    data = kfp.components.load_component_from_file('data/data.yaml')
    knn_train = kfp.components.load_component_from_file('knn_train/knn_train.yaml')
    knn_inference = kfp.components.load_component_from_file('knn_test/knn_test.yaml')

    # Get data and split in train and validation
    data_task = data("/cos/").add_pvolumes({"/cos/": data_op.volume})
    print(data_task.output)

    # Train and Test after splitting the data
    knn_train_task = knn_train(data_task.output, 20, 3).add_pvolumes({"/cos/": data_op.volume})
    print(knn_train_task.output)
    knn_inference_task = knn_inference( knn_train_task.output).add_pvolumes({"/cos/": data_op.volume})



if __name__ == '__main__':
    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'pipeline.yaml')
    # kfp.Client().create_run_from_pipeline_func(basic_pipeline, arguments={})
