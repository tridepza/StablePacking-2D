"""Pack a wall using the shapes and quantities configured in stone_types.

Each entry specifies a PNG in the data folder, its pixel label, and the
number of copies to make available. Use one entry to pack a single shape.
"""
import argparse
import os
import pathlib
import datetime
import random
import matplotlib
from skimage.measure import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np
from Stonepacker2D import *


def seed_everything(seed):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
parser.add_argument("--seed", type=int, help="Repeat a run with the same random seed.")
# The Windows runner also passes its own compatibility options.
args, _ = parser.parse_known_args()
seed = args.seed if args.seed is not None else random.SystemRandom().getrandbits(32)
if not 0 <= seed < 2**32:
    parser.error("--seed must be between 0 and 4294967295")
seed_everything(seed)
print(f"Run seed: {seed}", flush=True)
# ____________________________________________________________________________parameters
# %%
_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
_rescale = 0.35  # Fit the reference stone set to the packing grid.
wall_width = 36*6#216
wall_height = 36*6-12
set_world_size(wall_width+2,wall_height+3)
set_ksteps(0)
set_stabalize_method('none')
set_head_joint_state(True)
set_local_width(10)
set_dilation(20)
set_number_cpu(10)
set_placement_optimization("image_convolve")
set_record_detail({'RECORD_PLACEMENT_IMG':False,'RECORD_PLACEMENT_VALUE':False})

matplotlib.interactive(False)
time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(_root_dir+'/result_'+time_stamp+'/'):
    os.mkdir(_root_dir+'/result_'+time_stamp+'/')

# Each type has a filename, a pixel label, and an available quantity.
copies_per_shape = 1
stone_types = [
    ("stones_reference_segmented.png", label_value, copies_per_shape)
    for label_value in range(1, 68)
]

stones = []
initial_rotation_angles = dict()  # positive clockwise
for filename, label_value, quantity in stone_types:
    stone_file = os.path.join(_root_dir, "data", filename)
    for _ in range(quantity):
        stone = Stone()
        if not stone.from_labeled_img(label_value, stone_file, _rescale):
            raise ValueError(f"Could not load a stone with label {label_value} from {stone_file}")
        stone.id = len(stones)
        stone.cal_shape_factor()
        stones.append(stone)
        initial_rotation_angles[stone.id] = 0

# ______________________________WEIGHTS
iw = 1
kw = 1
cw = 1
gw = 1
lw = 1
sw = {'interlocking': iw, 'kinematic': kw,
    'course_height': cw, 'global_density': gw, 'local_void': lw}
# ______________________________END WEIGHTS
set_selection_weights(sw)
matplotlib.interactive(False)
set_dump(True, _root_dir+'/result_'+time_stamp+'/')
if not os.path.exists(_root_dir+'/result_'+time_stamp+'/'):
    os.mkdir(_root_dir+'/result_'+time_stamp+'/')
    os.mkdir(_root_dir+'/result_'+time_stamp+'/img/')
    os.mkdir(_root_dir+'/result_'+time_stamp+'/record/')
else:
    os.system('rm -rf '+_root_dir+'/result_'+time_stamp+'/*')
    os.mkdir(_root_dir+'/result_'+time_stamp+'/img/')
    os.mkdir(_root_dir+'/result_'+time_stamp+'/record/')
with open(get_dir()+'img/selection.txt', 'w+') as f:
    f.write("iteration;stone id;width;x;y;score;course height;global density;interlocking;local_density;kinematic\n")

base = Base()
ground = Brick(wall_width+2, 2)
ground.id = get_base_id()
base.add_stone(ground, [0, 0])
#base.matrix[:, -15:-1] = 1
base.matrix[:, -1] = get_ignore_pixel_value()
base.matrix[:, 0] = get_ignore_pixel_value()
base.matrix[-1, :] = get_ignore_pixel_value()
vendor = Vendor(stones)
nb_clusters = vendor.cluster_stones()
final_base = build_wall_vendor(
    base, vendor, 1,vendor_type = 'random')

# Output transformation
save_transformation_dir = get_dir()
with open(save_transformation_dir+'transformation.txt', 'w+') as f:
    f.write("id;d_x;d_y;angle\n")
for placed_stone in final_base.placed_rocks:
    stone_prop = regionprops(placed_stone.matrix.astype(np.uint8))
    stone_center = stone_prop[0].centroid
    d_x = (stone_center[1])/_rescale
    d_y = (stone_center[0])/_rescale
    angle_d = placed_stone.rot_angle
    angle_i = placed_stone.rotate_from_ori
    stone_id = placed_stone.id
    print(
        f"stone {stone_id} is placed at ({d_x},{d_y}) with stable angle {angle_d} degree")
    with open(save_transformation_dir+'transformation.txt', 'a+') as f:
        f.write("{};{};{};{}\n".format(placed_stone.id, d_x, d_y,
                initial_rotation_angles[placed_stone.id]+angle_i+angle_d))


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

plt.figure()
plt.imshow(final_base.matrix, cmap=newcmp,interpolation='none')
plt.axis('off')
text_kwargs = dict(ha='center', va='center',
                    fontsize=4, color='black')
for i, stone_i in enumerate(final_base.placed_rocks):

    plt.text(final_base.rock_centers[i][0],
                final_base.rock_centers[i][1], str(i+1), **text_kwargs)
plt.gca().invert_yaxis()
result_plt_name = _root_dir+'/result_'+time_stamp+'/' + \
    'Base_' + \
    str(sw['interlocking'])+'_'+str(sw['kinematic'])+'_' + \
    str(sw['course_height'])+'_'+str(sw['global_density'])+'.png'
plt.savefig(result_plt_name, dpi=300)

evaluate_kine(final_base, save_failure=True, load='tilting_table')

