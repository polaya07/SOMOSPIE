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
from generate_eval import *


@dsl.pipeline(name='somospie data generation pipeline', description='Pipeline for somospie data generation')
def pipeline(
    year: int = 2010, #Year to fetch soil moisture data.
    averaging_type: str = "monthly", #Averaging type (monthly, weekly, n_days).
    container_image: str ="icr.io/somospie/somospie-gdal-netcdf", 
    sm_name: str = "po-train", #'tn-sm', #"po-train"
    terrain_name: str = "ok-10m", #'tn-30m', #"oklahoma-30m", #"ok-10m",
    train_cos: str = "oklahoma-27km", #'tn-27km',#"oklahoma-27km"
    eval_cos: str = "ok-10m-eval", #tn-30m-eval, #"oklahoma-30m-eval", #ok-10m-eval
    shape_cos: str = "po-shapes",
    n_tiles: int = 6, #2, #6
    projection: str = 'EPSG:4326',
    region_res: str = "oklahoma-10m"#'tn-30m' #"oklahoma-30m" #"oklahoma-10m"
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

    #Eval matrices in tif format
    eval_pvc_op = dsl.VolumeOp(name="pvc-eval",
                           resource_name="pvc-eval",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": eval_cos, 
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

    ### Define region and resolution
    region_res: str = "ok-10m" #"tn-30m" #"ok-30m" 

    ### Get shape file to crop region for Training and Evaluation data
    shp_file: str = "OK.zip" #"TN.zip" #"OK.zip"
    get_shape_op = kfp.components.create_component_from_func(get_shp, base_image = str(container_image))
    get_shape_task=get_shape_op(shp_file, "/shape/").add_pvolumes({"/shape/": shape_pvc_op.volume}).set_retry(3)
   
    ### Training data
    ## Download data
    # The docker container uses `multiprocess` that does not require top-level functions. I need to push this container
    # to ibm container registry but i exceeded monthly quota (2 images per month) 
    download_sm_op = kfp.components.create_component_from_func(download_parallel, base_image = "olayap/somospie-gdal-netcdf")#str(container_image))#"olayap/somospie-gdal-netcdf")
    average_rasters_op = kfp.components.create_component_from_func(merge_avg, base_image = str(container_image))
    
    
    sm_avg_files = [("/sm/"+str(year)+'/{0:02d}.tif'.format(month)) for month in range(1, 13)] #In the NVME!
    
    download_sm_task=download_sm_op(year, "/sm/").add_pvolumes({"/sm/": sm_pvc_op.volume}).set_retry(3)

    sm_files=[]
    # The output of these ones can be intermediate data! (.tif)
    if averaging_type == 'monthly':
        for month in range(1, 13):
            sm_files.append(average_rasters_op(download_sm_task.output,year, month, sm_avg_files[month - 1], projection).add_pvolumes({"/sm/": sm_pvc_op.volume}).set_retry(3))
            # break

    ## Generate train data: Combine soil moisture data with the terrain parameters
    build_stack_train_op = kfp.components.create_component_from_func(build_stack_train, base_image = str(container_image))
    crop_region_train_op = kfp.components.create_component_from_func(crop_region_train, base_image = str(container_image))

    param_names = ['aspect', 'elevation', 'hillshading', 'slope']
    terrain_params = ["/terrain/"+terrain+"_reprojected.tif" for terrain in param_names]
    year=2010
    
    for i, sm_avg_file in enumerate(sm_avg_files):
        print(i, sm_avg_file)
        train_file = ("/train/"+'{0:04d}_{1:02d}.tif'.format(year, i + 1))
        print(train_file)
        #Build stack: Get soil moisture and terrain parameters to build the stack
        build_stack_train_task=build_stack_train_op("/train/", sm_files[i].output, terrain_params, train_file, year, i+1).add_pvolumes({"/sm/": sm_pvc_op.volume}).add_pvolumes({"/terrain/": terrain_pvc_op.volume}).add_pvolumes({"/train/": train_pvc_op.volume}).set_memory_request('1G').set_retry(3)
        #Crop per region: def crop_region(input_file:str, zip_file:str, output_file:str, parameter_names:list)
        crop_region_train_op(build_stack_train_task.output, get_shape_task.output, train_file, param_names, year, i+1).add_pvolumes({"/train/": train_pvc_op.volume}).add_pvolumes({"/shape/": shape_pvc_op.volume}).set_memory_request('1G').set_retry(3)

   
    
    #### Evaluation data
    # Create components
    build_stack_eval_op = kfp.components.create_component_from_func(build_stack_eval, base_image = str(container_image))
    crop_region_eval_op = kfp.components.create_component_from_func(crop_region_eval, base_image = str(container_image))
    crop_tile_op = kfp.components.create_component_from_func(crop_tile, base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    write_stack_op = kfp.components.create_component_from_func(write_stack, base_image = str(container_image))
    band_names_op = kfp.components.create_component_from_func(band_names, base_image = str(container_image))

    param_names = ['aspect', 'elevation', 'hillshading', 'slope']

    # Crop terrain params for the specific region
    cropped_files=[]
    for i,f in enumerate(terrain_params):
        cropped_terrain="/terrain/"+param_names[i]+"_cropped.tif"
        cropped_files.append(crop_region_eval_op(f, get_shape_task.output, cropped_terrain).add_pvolumes({"/terrain/": terrain_pvc_op.volume}).add_pvolumes({"/shape/": shape_pvc_op.volume}).set_retry(3))   

    build_stack_eval_task=build_stack_eval_op([cropped.output for cropped in cropped_files], "/terrain/stack.vrt").add_pvolumes({"/terrain/": terrain_pvc_op.volume}).set_memory_request('1G').set_retry(3)

    n_tiles = 6 #2 #6 
    tile_count = 0 

    if n_tiles == 0:
        eval_file = '/eval/'+region_res+'_eval.tif'
        write_stack_task=write_stack_op(build_stack_eval_task.output, eval_file).add_pvolumes({"/terrain/": terrain_pvc_op.volume}).add_pvolumes({"/eval/": eval_pvc_op.volume}).set_retry(3)
        band_names_op(write_stack_task.output,param_names).add_pvolumes({"/eval/": eval_pvc_op.volume}).set_retry(3)
        #clean to delete "/terrain/stack.vrt" ...
    else:
        for i in range(n_tiles):
            for j in range(n_tiles):
                eval_file = '/eval/'+region_res+'_eval_{0:04d}.tif'.format(tile_count)
                crop_tile_task=crop_tile_op(build_stack_eval_task.output, eval_file, n_tiles, i, j).add_pvolumes({"/eval/": eval_pvc_op.volume}).add_pvolumes({"/terrain/": terrain_pvc_op.volume}).set_memory_request('1G').set_retry(3)
                band_names_op(crop_tile_task.output,param_names).add_pvolumes({"/eval/": eval_pvc_op.volume}).set_memory_request('1G').set_retry(3)
                tile_count += 1
    #clean to delete "/terrain/stack.vrt" ...        

if __name__ == '__main__':

    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'data-transformation-pipeline/pipeline.yaml')