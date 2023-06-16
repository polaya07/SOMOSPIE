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
    links_file: str ="/cos/TN_30m.txt", 
    projection_file: str ="/cos/albers_conus_reference.wkt",
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
    reproject_task = reproject_op(merge_task.output, "/cos/dem.tif", projection_file).add_pvolumes({"/cos/": pvc_op.volume})

    # Define terrain parameters
    aspect_tiles = []
    hillshading_tiles = []
    slope_tiles = []
    tile_count = 0
    n_tiles=2
    for i in range(n_tiles):
        for j in range(n_tiles):
            tile = "tile_{0:04d}.tif"
            aspect_tiles.append("aspect_tile_{0:04d}.tif".format(tile_count))
            hillshading_tiles.append("hillshading_tile_{0:04d}.tif".format(tile_count))
            slope_tiles.append("slope_tile_{0:04d}.tif".format(tile_count))

            # Crop tile
            crop_task = crop_op(reproject_task.output, "/cos/"+tile, n_tiles, i, j).add_pvolumes({"/cos/": pvc_op.volume})

            # Compute tile
            compute_task = compute_op(crop_task.output, "/cos/"+aspect_tiles[-1], "/cos/"+hillshading_tiles[-1], "/cos/"+slope_tiles[-1]).add_pvolumes({"/cos/": pvc_op.volume})
            
            tile_count += 1

    # Merge all tiles for all terrain parameters

    

if __name__ == '__main__':

    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'data-generation-pipeline/pipeline.yaml')