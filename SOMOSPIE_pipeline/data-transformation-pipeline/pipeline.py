import kfp
from kfp import dsl 
#from kfp.v2 import dsl as dsl2
from kfp.components import func_to_container_op
from typing import NamedTuple
import os


# Import functions from other files
from build_stack import *
from tif2df import *


@dsl.pipeline(name='somospie data generation pipeline', description='Pipeline for somospie data generation')
def pipeline(
    container_image: str ="olayap/somospie-gdal", 
    #links_file: str ="/cos/TN_30m.txt", 
    #projection_file: str ="/cos/albers_conus_reference.wkt",
    ):

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

    # Create components
    build_stack_op = kfp.components.create_component_from_func(build_stack, base_image = str(container_image))
    tif2df_op = kfp.components.create_component_from_func(tif2df, base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])

    #Build stack
     #input_files:list, output_file:str)->str:
    tifs=['dem.tif','aspect.tif','hillshading.tif','slope_tiles.tif']
    tifs=["/cos/" + s for s in tifs]
    build_stack_task = build_stack_op(tifs,"/cos/terrain-params.tif").add_pvolumes({"/cos/": pvc_op.volume})
    tif2df_task = tif2df_op( build_stack_task.output,['elevation'],'params.csv').add_pvolumes({"/cos/": pvc_op.volume})
    

if __name__ == '__main__':

    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'data-transformation-pipeline/pipeline.yaml')