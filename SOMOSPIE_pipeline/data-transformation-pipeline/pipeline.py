import kfp
from kfp import dsl 
from kfp.components import func_to_container_op
from typing import NamedTuple
import os
import glob
import calendar


# Import functions from other files
from get_sm import *
from generate_train import *


@dsl.pipeline(name='somospie data generation pipeline', description='Pipeline for somospie data generation')
def pipeline(
    year: int = 2010, #Year to fetch soil moisture data.
    averaging_type: str = "monthly", #Averaging type (monthly, weekly, n_days).
    container_image: str ="icr.io/somospie/somospie-gdal-netcdf", 
    sm_name: str = "po-train",
    terrain_name: str = "oklahoma-30m",
    train_cos: str = "oklahoma-27km", 
    shape_cos: str = "po-shapes",
    n_tiles: int = 3,
    projection: str = 'EPSG:4326',
    ):

    # Create a PVC where input data is stored

    #Soil Moisture data
    sm_pvc_op = dsl.VolumeOp(name="pvc-sm",
                           resource_name="pvc-sm",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": sm_name, 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO).set_retry(3)

    #Terrain parameters tifs
    terrain_pvc_op = dsl.VolumeOp(name="pvc-terrain",
                           resource_name="pvc-terrain",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": terrain_name, 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO).set_retry(3)
    
    #Train matrices in tif format
    train_pvc_op = dsl.VolumeOp(name="pvc-train",
                           resource_name="pvc-train",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": train_cos, 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO).set_retry(3)

    #Shape files
    shape_pvc_op = dsl.VolumeOp(name="pvc-shape",
                           resource_name="pvc-shape",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": shape_cos, 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO).set_retry(3)


    ### Training data
    ## Download data
    download_sm_op = kfp.components.create_component_from_func(download, base_image = str(container_image))
    average_rasters_op = kfp.components.create_component_from_func(merge_avg, base_image = str(container_image))
    
    
    sm_avg_files = [("/sm/"+str(year)+'/{0:02d}.tif'.format(month)) for month in range(1, 13)] #In the NVME!
    
    download_sm_task=download_sm_op(year, "/sm/").add_pvolumes({"/sm/": sm_pvc_op.volume})

    sm_files=[]
    # The output of these ones can be intermediate data! (.tif)
    if averaging_type == 'monthly':
        for month in range(1, 13):
            sm_files.append(average_rasters_op(download_sm_task.output,year, month, sm_avg_files[month - 1], projection).add_pvolumes({"/sm/": sm_pvc_op.volume}).set_retry(3))
            # break

    ## Generate train data: Combine soil moisture data with the terrain parameters
    build_stack_op = kfp.components.create_component_from_func(build_stack, base_image = str(container_image))
    crop_region_op = kfp.components.create_component_from_func(crop_region, base_image = str(container_image))

    param_names = ['aspect', 'elevation', 'hillshading', 'slope']
    terrain_params = ["/terrain/"+terrain+".tif" for terrain in param_names]
    year=2010
    shp_file: str = "OK.zip"
    for i, sm_avg_file in enumerate(sm_avg_files):
        print(i, sm_avg_file)
        train_file = ("/train/"+'{0:04d}_{1:02d}.tif'.format(year, i + 1))
        print(train_file)
        #Build stack: Get soil moisture and terrain parameters to build the stack
        build_stack_task=build_stack_op("/train/", sm_files[i].output, terrain_params, train_file, year, i+1).add_pvolumes({"/sm/": sm_pvc_op.volume}).add_pvolumes({"/terrain/": terrain_pvc_op.volume}).add_pvolumes({"/train/": train_pvc_op.volume}).set_retry(3)
        #Crop per region: def crop_region(input_file:str, zip_file:str, output_file:str, parameter_names:list)
        crop_region_op(build_stack_task.output, shp_file, "/shape/", train_file, param_names, year, i+1).add_pvolumes({"/train/": train_pvc_op.volume}).add_pvolumes({"/shape/": shape_pvc_op.volume}).set_retry(3)

   
    
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