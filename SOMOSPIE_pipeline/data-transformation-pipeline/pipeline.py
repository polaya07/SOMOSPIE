import kfp
from kfp import dsl 
from kfp.components import func_to_container_op
from typing import NamedTuple
import os
import glob
import calendar


# Import functions from other files
from get_sm import *
from build_stack import *
from crop_tile import *
from write_stack import *


@dsl.pipeline(name='somospie data generation pipeline', description='Pipeline for somospie data generation')
def pipeline(
    year: int = 2010, #Year to fetch soil moisture data.
    averaging_type: str = "monthly", #Averaging type (monthly, weekly, n_days).
    container_image: str ="icr.io/somospie/somospie-gdal-netcdf", 
    cos_name: str = "po-train",
    n_tiles: int = 3,
    projection: str = 'EPSG:4326',
    ):

    # Create a PVC where input data is stored
    pvc_op = dsl.VolumeOp(name="pvc-train",
                           resource_name="pvc-train",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": cos_name, 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO)


    ### Training data
    ## Download data
    download_sm_op = kfp.components.create_component_from_func(download, base_image = str(container_image))
    average_rasters_op = kfp.components.create_component_from_func(merge_avg, base_image = str(container_image))
    
    
    output_files = [("/cos/"+'{0:04d}/{0:02d}.tif'.format(year, month)) for month in range(1, 13)] #In the NVME!
    
    download_sm_task=download_sm_op(year, "/cos/").add_pvolumes({"/cos/": pvc_op.volume})

    # The output of these ones can be intermediate data! (.tif)
    if averaging_type == 'monthly':
        for month in range(1, 13):
            average_rasters_op(download_sm_task.output,year, month, output_files[month - 1], projection).add_pvolumes({"/cos/": pvc_op.volume}).set_retry(3)
            # break

    
"""     #### Evaluation data
    # Create components
    build_stack_op = kfp.components.create_component_from_func(build_stack, base_image = str(container_image))
    crop_tile_op = kfp.components.create_component_from_func(crop_tile, base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    write_stack_op = kfp.components.create_component_from_func(write_stack, base_image = str(container_image))

    #Build stack
     #input_files:list, output_file:str)->str:
    tifs=['dem.tif','aspect.tif','hillshading.tif','slope_tiles.tif']
    tifs=["/cos/" + s for s in tifs]
    build_stack_task = build_stack_op(tifs,"/cos/terrain-params.tif").add_pvolumes({"/cos/": pvc_op.volume})

    n_tiles = 3 
    tile_count = 0 

    if n_tiles == 0:
        write_stack_op(build_stack_task.output, output_file).add_pvolumes({"/cos/": pvc_op.volume})
    else:
        for i in range(n_tiles):
            for j in range(n_tiles):
                tile = "eval_tile_{0:04d}.tif".format(tile_count)
                crop_tile_task = crop_tile_op(build_stack_task.output, "/cos/"+tile, n_tiles, i, j).add_pvolumes({"/cos/": pvc_op.volume}).set_retry(3)
                tile_count += 1 """
        

if __name__ == '__main__':

    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'data-transformation-pipeline/pipeline.yaml')