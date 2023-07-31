import kfp
from kfp import dsl 
from kfp.components import func_to_container_op
from typing import NamedTuple
import os


# Import functions from other files
from load_dems import *
from merge import *
from reproject import *
from crop import *
from compute import *
from merge_avg import *

@dsl.pipeline(name='somospie data generation pipeline', description='Pipeline for somospie data generation')
def pipeline(
    container_image: str ="olayap/somospie-gdal", 
    links_file: str ="/cos/OK_10m.txt", #"/cos/OK_30m.txt",
    n_tiles: int = 3, #2 #3
    projection_file: str ="/cos/albers_conus_reference.wkt", 
    cos_name: str = "ok-10m" #"ok-10m" #"tn-30m" #"oklahoma-30m"
    ): 

    #set_retry_op = dsl.set_retry(num_retries=3)
    # Create a PVC where input data is stored
    pvc_op = dsl.VolumeOp(name="pvc-geotiled",
                           resource_name="pvc-goetiled",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": cos_name, 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO)



    # How to define a component: https://kubeflow-pipelines.readthedocs.io/en/stable/source/kfp.components.html#kfp.components.create_component_from_func
    load_data_op = kfp.components.create_component_from_func(load_data, base_image = str(container_image))
    merge_op = kfp.components.create_component_from_func(merge_tiles, base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    reproject_op = kfp.components.create_component_from_func(reproject,  base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    crop_op = kfp.components.create_component_from_func(crop_into_tiles,  base_image = str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])
    compute_op = kfp.components.create_component_from_func(compute_geotiled,  base_image =str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])     
    merge_avg_op = kfp.components.create_component_from_func(merge_avg,  base_image =str(container_image))#packages_to_install=["numpy", "pandas", "scikit-learn"])     


     # Get data and split in train and validation
    data_task = load_data_op(links_file,"/cos/dems/").add_pvolumes({"/cos/": pvc_op.volume})
    # Merge
    merge_task = merge_op(data_task.output, "/cos/mosaic.tif", "/cos/").add_pvolumes({"/cos/": pvc_op.volume})
    # Reproject
    reproject_task = reproject_op(merge_task.output, "/cos/elevation_m.tif", projection_file, 'n').add_pvolumes({"/cos/": pvc_op.volume})

    # Define terrain parameters
    aspect_tiles = []
    hillshading_tiles = []
    slope_tiles = []
    compute_task = []
    tile_count = 0
    n_tiles=3
    #with dsl.ParallelFor(list(range(n_tiles))) as i:
    #    with dsl.ParallelFor(list(range(n_tiles))) as j:
    #print(container_image)
    #print(vars(n_tiles))
    for i in range(n_tiles):
        for j in range(n_tiles):
            tile = "tile_{0:04d}.tif".format(tile_count)
            aspect_tiles.append("aspect_tile_{0:04d}.tif".format(tile_count))
            hillshading_tiles.append("hillshading_tile_{0:04d}.tif".format(tile_count))
            slope_tiles.append("slope_tile_{0:04d}.tif".format(tile_count))

            # Crop tile
            #dsl.set_retry(num_retries= 3)
            crop_task = crop_op(reproject_task.output, "/cos/"+tile, n_tiles, i, j).add_pvolumes({"/cos/": pvc_op.volume}).set_retry(3)

            # Compute tile
            compute_task.append(compute_op("/cos/", tile_count, crop_task.output, "/cos/"+aspect_tiles[-1], "/cos/"+hillshading_tiles[-1], "/cos/"+slope_tiles[-1]).add_pvolumes({"/cos/": pvc_op.volume}).set_retry(3))
            tile_count += 1

    reproject_task_elevation = reproject_op(merge_task.output, "/cos/elevation_reprojected.tif", 'EPSG:4326', 'y', compute_task[0].outputs).add_pvolumes({"/cos/": pvc_op.volume}).set_retry(3)

    # Merge all tiles for all terrain parameters
    merge_avg_task_aspect = merge_avg_op([compute_task[z].outputs['aspect'] for z in range(tile_count)],"/cos/", "/cos/aspect.tif", "aspect").add_pvolumes({"/cos/": pvc_op.volume}).set_memory_request('15G').set_retry(3)
    merge_avg_task_hill = merge_avg_op([compute_task[z].outputs['hill'] for z in range(tile_count)],"/cos/", "/cos/hillshading.tif", "hillshading").add_pvolumes({"/cos/": pvc_op.volume}).set_memory_request('15G').set_retry(3)
    merge_avg_task_slope = merge_avg_op([compute_task[z].outputs['slope'] for z in range(tile_count)], "/cos/","/cos/slope.tif", "slope").add_pvolumes({"/cos/": pvc_op.volume}).set_memory_request('15G').set_retry(3)
    
    # Reproject 
    reproject_task_aspect = reproject_op(merge_avg_task_aspect.output, "/cos/aspect_reprojected.tif", 'EPSG:4326', 'n').add_pvolumes({"/cos/": pvc_op.volume}).set_retry(3)
    reproject_task_hill = reproject_op(merge_avg_task_hill.output, "/cos/hillshading_reprojected.tif", 'EPSG:4326', 'n').add_pvolumes({"/cos/": pvc_op.volume}).set_retry(3)
    reproject_task_slope = reproject_op(merge_avg_task_slope.output, "/cos/slope_reprojected.tif", 'EPSG:4326', 'n').add_pvolumes({"/cos/": pvc_op.volume}).set_retry(3)


if __name__ == '__main__':

    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'data-generation-pipeline/pipeline.yaml')